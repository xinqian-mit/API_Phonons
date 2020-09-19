#!/usr/bin/env python3
# coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from phonopy import Phonopy
import phonopy.interface.vasp as Intf_vasp
from phonopy.structure.atoms import PhonopyAtoms
import phonopy.file_IO as PhonIO
from phonopy.interface.calculator import get_default_physical_units
import API_quippy_phonopy_VASP as api_qpv # remember to set this module to python path
import copy as cp
#from phonopy.interface.alm import get_fc2

import API_alamode as api_alm
import API_quippy_thirdorder as shengfc3
import thirdorder_core
import thirdorder_common


Temperatures= [50,100,150,200,300]
NSnaps = 50 # number of snapshots generated per sc loop.
Niter = 8 # number of iterations
nneigh = 3 # nearest neighbor cutoff for fc3.

if_write_iter = False # whether write sc iteratoins
gp_xml_file= "../Dielectric_function_NaCl/soap_n12l11_6.0cut_coul/gp_NaCl_soap_coul.xml"
Qpoints=np.array([[0.0001,0.0001,1.0],[0.5,0.5,1.0],[3./8,3./8,3./4],[0.0,0.0,0.0],[0.5,0.5,0.5]]) #BZ path
band_labels=['$\Gamma$','X','K','$\Gamma$','L']

Ncells=[4,4,4] # supcer cells
Band_points=51 # points on each BZ-path segments
NAC = True # non-analytic correction for polar solids
interface_mode = 'vasp' # interface mode. use vasp.

Unit_cell = Intf_vasp.read_vasp("POSCAR") # read prim cell from the POSCAR file
prim_mat = np.eye(3)#[[0, 0.5, 0.5],[0.5, 0, 0.5],[0.5, 0.5, 0]]


# Obtain Eigenvectors using Finite Displace
phonon = Phonopy(Unit_cell,np.diag(Ncells),primitive_matrix=prim_mat)
phonon_scell = Phonopy(phonon.get_supercell(),np.eye(3),primitive_matrix=prim_mat) # generate an phononpy object for LD calc.
phonon_scell.generate_displacements(distance=0.03) # vasp
Scells_phonopy = phonon_scell.get_supercells_with_displacements() # This returns a list of Phononpy atoms object



Scells_quippy=[]
for scell in Scells_phonopy:
    Scells_quippy.append(api_qpv.phonopyAtoms_to_aseAtoms(scell))



force_quip_scells = api_qpv.calc_force_sets_GAP(gp_xml_file,Scells_quippy)

#parse force set and calc force constants
phonon_scell.set_forces(force_quip_scells)
PhonIO.write_FORCE_SETS(phonon_scell.get_displacement_dataset()) # write forces & displacements to FORCE_SET
force_set=PhonIO.parse_FORCE_SETS() # parse force_sets
phonon_scell.set_displacement_dataset(force_set) # force_set is a list of forces and displacements


phonon_scell.produce_force_constants()
phonon_scell.symmetrize_force_constants()

if NAC == True:
    nac_params_sc = PhonIO.get_born_parameters(open("BORN"), 
                                                phonon_scell.get_primitive(),
                                                phonon_scell.get_primitive_symmetry())
    if nac_params_sc['factor'] == None:
        physical_units = get_default_physical_units(interface_mode)
        nac_params_sc['factor'] = physical_units['nac_factor']
    phonon_scell._nac_params = nac_params_sc
    
phonon_scell._set_dynamical_matrix()
DynMat0 = phonon_scell._dynamical_matrix.get_dynamical_matrix()
bands_sc=api_qpv.qpoints_Band_paths([[0.0,0.0,0.0]],1) #supercell
phonon_scell.set_band_structure(bands_sc, is_eigenvectors=True)
phonon_scell.set_mesh([1,1,1], is_eigenvectors=True)

# Get frange for third order force constants:
poscar = shengfc3.read_POSCAR(".")
sposcar = shengfc3.gen_SPOSCAR(poscar, Ncells[0], Ncells[1], Ncells[2])
dmin, nequi, shifts = shengfc3.calc_dists(sposcar)
frange = shengfc3.calc_frange(poscar, sposcar, nneigh, dmin)*10 # get cutoff in Angstrom.
options ='solver = dense, cutoff = '+str(frange)

Natoms = phonon.get_supercell().get_number_of_atoms()
Supercell = phonon_scell.get_supercell()
Intf_vasp.write_vasp("SPOSCAR_NaCl",Supercell)
pos0 = Supercell.get_positions()

# Generate thermal displacements along eigenvectors, i.e. populate phonon modes.

for Temperature in Temperatures:
    FC2Sum = np.zeros(phonon_scell.get_force_constants().shape)
    FC2ave = np.zeros(phonon_scell.get_force_constants().shape)
    FC3Sum = np.zeros([Natoms,Natoms,Natoms,3,3,3])
    FC3ave = np.zeros([Natoms,Natoms,Natoms,3,3,3])
    for icalc in range(Niter):
        u_disps = api_qpv.thermo_disp_along_eig(phonon_scell,Temperature,NSnaps)
        Scell_snaps = [];
        
        
        snaps_ase=[]
        for isnap in range(NSnaps):
            Scell_tmp = cp.deepcopy(Supercell)
            pos = pos0 + u_disps[isnap]
            Scell_tmp.set_positions(pos)
            Scell_snaps.append(Scell_tmp)
            snaps_ase.append(api_qpv.phonopyAtoms_to_aseAtoms(Scell_tmp))
        
        displacements,forces=api_qpv.get_DFSETS_GAP(Supercell,Scell_snaps,gp_xml_file) # in API_phonopy_lammps, there's a corresponding get_DFSETS_lmp for empirical potential.
        #FC2=get_fc2(Supercell,phonon.get_primitive(),displacements,forces,log_level=1)
        FC2,FC3 = api_alm.get_fc2_fc3(phonon,displacements,forces,is_compact_fc=False,options=options,log_level=1)
        FC2Sum = FC2Sum + FC2
        FC2ave = FC2Sum/(icalc+1)
        FC3Sum = FC3Sum + FC3
        FC3ave = FC3Sum/(icalc+1)
        phonon_scell.set_force_constants(FC2ave) # average.
        bands_sc=api_qpv.qpoints_Band_paths([[0.0,0.0,0.0]],1) #supercell
        phonon_scell.set_band_structure(bands_sc, is_eigenvectors=True)
        phonon_scell.set_mesh([1,1,1], is_eigenvectors=True)
        
        phonon.set_force_constants(FC2ave)
        phonon.symmetrize_force_constants()
        
        
        if NAC == True:
            nac_params = PhonIO.get_born_parameters(open("BORN"), 
                                                    phonon.get_primitive(),
                                                    phonon.get_primitive_symmetry())
            nac_params_sc = PhonIO.get_born_parameters(open("BORN"), 
                                                    phonon_scell.get_primitive(),
                                                    phonon_scell.get_primitive_symmetry())
                
            if nac_params['factor'] == None:
                physical_units = get_default_physical_units(interface_mode)
                nac_params['factor'] = physical_units['nac_factor']
            if nac_params_sc['factor'] == None:
                nac_params_sc['factor'] = physical_units['nac_factor']
            phonon._nac_params = nac_params
            phonon_scell._nac_params = nac_params_sc
            phonon._set_dynamical_matrix()
            phonon_scell._set_dynamical_matrix()
        
        if if_write_iter:
            bands=api_qpv.qpoints_Band_paths(Qpoints,Band_points)
            phonon.set_band_structure(bands,labels=band_labels)
            bs_plt=phonon.plot_band_structure()
            phonon.write_yaml_band_structure(filename='NaCl_disp_'+str(Temperature)+'K_iter'+str(icalc+1)+'.yaml')

    
    PhonIO.write_FORCE_CONSTANTS(phonon.get_force_constants(), filename='FORCE_CONSTANTS_'+str(Temperature)+'K')
    bands=api_qpv.qpoints_Band_paths(Qpoints,Band_points)
    phonon.set_band_structure(bands,labels=band_labels)
    bs_plt=phonon.plot_band_structure()
    phonon.write_yaml_band_structure(filename='NaCl_disp_'+str(Temperature)+'K.yaml')
    
    prim = api_qpv.phonopyAtoms_to_aseAtoms(phonon.get_primitive())
    api_alm.write_shengBTE_fc3('FORCE_CONSTANTS_3RD_'+str(Temperature)+'K',FC3ave,phonon,prim)





