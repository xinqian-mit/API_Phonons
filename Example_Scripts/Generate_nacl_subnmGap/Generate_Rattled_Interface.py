#!/usr/bin/env python3

from ase.spacegroup import crystal
from ase.visualize import view
from ase.io import write
import ase.build
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
import API_phonopy as api_ph
import API_quippy as api_q
import API_phonopy_lammps as api_plmp
from phonopy import Phonopy
import phonopy.file_IO as PhonIO
from phonopy.interface.calculator import get_default_physical_units
import copy as cp
import multiprocessing as mp
import phonopy.interface.vasp as phonVasp


# Specify the slab cut and layers
NAC = True # non-analytical correction for LO-TO splitting
gp_xml_file = '../Dielectric_function_NaCl/soap_n12l11_6.0cut_coul/gp_NaCl_soap_coul.xml'

a = 5.65414946767 # lattice constant of NaCl

Temp_rattle = [300.,300]
Nsnaps = 2 #50000 # Number of snapshots
nacl = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
               cellpar=[a, a, a, 90, 90, 90])


d0 = a/2.0;
d_gap = 0  # Angstrom, distance of the gap
d_vac = 60
Stacking_Period = 2 # Period of stacking. In (111) stacking, the layer # is 6. In (110) stacking, the layer # is 4
Nrepeat = (2,3,4) # for lattice dynamics on the unit cell.

#Cutvec_a = (1,-1,0); Cutvec_b = (1,1,-2) # cut in (111) direction, Period of stacking in (111) is 6 layer
Cutvec_a = (1,-1,0); Cutvec_b = (0,0,1) # cut in (110) direction
facedir = np.abs(np.cross(np.array(Cutvec_a),np.array(Cutvec_b)))
facestr = str(facedir[0])+str(facedir[1])+str(facedir[2])

uc_slab = ase.build.cut(nacl, a=Cutvec_a, b=Cutvec_b,nlayers=Stacking_Period)
print('Number of atoms in the stacking unit cell: '+ str(uc_slab.get_global_number_of_atoms()))

# Generate interface
slab = uc_slab.repeat(Nrepeat)
Interface0 = ase.build.stack(slab,slab,maxstrain=100,distance=d_gap+d0)
Slab_vecs = slab.get_cell()
Thick_slab = np.linalg.norm(Slab_vecs[2,:])
ase.build.rotate(Interface0, Slab_vecs[0,:],[1,0,0], Slab_vecs[1,:], [0,1,0], rotate_cell=True)
#Interface0.translate([0,0,d_gap/2.])
Interface0.set_velocities(np.zeros(3))
Interface0.center(vacuum=d_vac,axis=2)

# set mol ID for two slabs.
molID = np.zeros(Interface0.get_global_number_of_atoms(),dtype=int)
molID[0:slab.get_global_number_of_atoms()] = 1
molID[slab.get_global_number_of_atoms():Interface0.get_global_number_of_atoms()] = 2
#view(Interface0)
ase.io.write('Interface0_'+facestr+'_'+str(d_gap)+'A.cif',Interface0,'cif')

Interface0.set_pbc([True,True,False])
lmp_Data_filename = 'Interface0_'+facestr+'_'+str(d_gap)+'A.data'
api_plmp.write_lmp_data(lmp_Data_filename,Interface0,molID)
phonVasp.write_vasp('Interface0_'+facestr+'_'+str(d_gap)+'A.POSCAR',Interface0)

if Nsnaps > 0:

    slab_ph = api_ph.aseAtoms_to_phonopyAtoms(slab)

    phonon = Phonopy(slab_ph,np.eye(3)) # do the Gamma point calculation of the unit prim cell
    phonon.generate_displacements(distance=0.01) # perturbate the supcercell to obtain eigenvectors.
    Scell_disps = phonon.get_supercells_with_displacements()
    Scell_ph = phonon.get_supercell()
    Scell_disps_quip = []
    for scell in Scell_disps:
        Scell_disps_quip.append(api_ph.phonopyAtoms_to_aseAtoms(scell)) # pass the phonopy atom to ase object.
    print('Supercell used for Lattice Dynamics: '+str(Nrepeat))
    print('Number of atoms in the slab: '+str(Scell_ph.get_number_of_atoms()))
    print('Number of Supercell calculations: '+str(len(Scell_disps)))
    # Calculate forces
    force_quip = api_q.calc_force_sets_GAP(gp_xml_file,Scell_disps_quip)
    phonon.set_forces(force_quip)
    PhonIO.write_FORCE_SETS(phonon.get_displacement_dataset()) # write forces & displacements to FORCE_SET
    force_set=PhonIO.parse_FORCE_SETS() # parse force_sets
    phonon.set_displacement_dataset(force_set) # force_set is a list of forces and displacements
    # Calculate force constants with or without NAC.
    if NAC == True:
        nac_params = PhonIO.get_born_parameters(
                open("BORN"),
                phonon.get_primitive(),
                phonon.get_primitive_symmetry())
        if nac_params['factor'] == None:
            physical_units = get_default_physical_units('vasp')
            nac_params['factor'] = physical_units['nac_factor']
        phonon._nac_params=nac_params
    phonon.produce_force_constants()


# calculate at Gamma point
    bands=api_ph.qpoints_Band_paths(np.array([[0.,0.,0.]]),1)
    phonon.set_band_structure(bands, is_eigenvectors=True)

    u_disps1,v_disps1 = api_ph.Parallel_thermo_dispVel_along_eig(phonon,Temp_rattle[0],Nsnaps) # rattle slab1
    u_disps2,v_disps2 = api_ph.Parallel_thermo_dispVel_along_eig(phonon,Temp_rattle[1],Nsnaps) # rattle slab2

    u_disps = np.concatenate((u_disps1,u_disps2),axis=1)
    v_disps = np.concatenate((v_disps1,v_disps2),axis=1)

    Interface_snaps = api_ph.Generate_Supercells_with_Disps(Interface0,u_disps,v_disps)
    api_ph.write_Supercells_VASP(Interface_snaps,'d_gap'+str(d_gap)+'A')

