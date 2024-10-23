#!/usr/bin/env python3
# coding: utf-8
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from phonopy import Phonopy
import phonopy.interface.vasp as Intf_vasp
from phonopy.structure.atoms import PhonopyAtoms
import phonopy.file_IO as PhonIO
from phonopy.interface.calculator import get_default_physical_units
import API_phonopy as api_ph # remember to set this module to python path
import API_quippy as api_q # remember to set this module to python path
import copy as cp
# set-up latex 


# specify potential here
gp_xml_file="../Dispersion_Zr_hcp_gap/gp_new_DB2-4.xml"

# perform a Gamma point LD on the cell specified by POSCAR
Qpoints=np.array([[0.0,0.0,0.0]])
Ncells=[3,3,2]
qmesh=[1,1,1]
Band_points=1
interface_mode = 'vasp'
NSnaps = 50
Temperature = 300
directory = 'Zr_hcp_Thermo_Disps_T'+str(Temperature)
Snap_file = 'Zr_hcp_Thermo_Disps_T'+str(Temperature)+'.xyz'


# generate displacements
Unit_cell = Intf_vasp.read_vasp("POSCAR_Zr") # read prim cell from the POSCAR file
prim_mat = [[1,0,0],[0,1,0],[0,0,1]] #[[0, 0.5, 0.5],[0.5, 0, 0.5],[0.5, 0.5, 0]]
phonon_scell = Phonopy(Unit_cell,np.diag(Ncells),primitive_matrix=prim_mat) # generate an phononpy object for LD calc.
phonon_scell.generate_displacements(distance=0.01) # vasp
Scells_phonopy = phonon_scell.get_supercells_with_displacements() # This returns a list of Phononpy atoms object

# convert phonopy atoms objects to quippy atom objects
Scells_quippy=[]
for scell in Scells_phonopy:
    Scells_quippy.append(api_ph.phonopyAtoms_to_aseAtoms(scell))


# calculate forces and convert to phonopy force_sets
#force_quip_scells = api_q.calc_force_sets_quip(pot_flag,param_str,Scells_quippy)
force_quip_scells = api_q.calc_force_sets_GAP(gp_xml_file,Scells_quippy)


#parse force set and calc force constants
phonon_scell.set_forces(force_quip_scells)
PhonIO.write_FORCE_SETS(phonon_scell.get_displacement_dataset()) # write forces & displacements to FORCE_SET
force_set=PhonIO.parse_FORCE_SETS() # parse force_sets
phonon_scell.set_displacement_dataset(force_set) # force_set is a list of forces and displacements


phonon_scell.produce_force_constants()
phonon_scell.symmetrize_force_constants()


# set qpoints along BZ path
bands=api_ph.qpoints_Band_paths(Qpoints,Band_points)
phonon_scell.set_band_structure(bands, is_eigenvectors=True)
phonon_scell.set_mesh(qmesh, is_eigenvectors=True)
phonon_scell.write_yaml_band_structure()
eigvecs=api_ph.get_reshaped_eigvecs(phonon_scell)



u_disps = api_ph.thermo_disp_along_eig(phonon_scell,Temperature,NSnaps)
Scell_snaps = [];
Supercell = phonon_scell.get_supercell()
Intf_vasp.write_vasp("SPOSCAR_Zr",Supercell)
pos0 = Supercell.get_positions()

snaps_ase=[]
for isnap in range(NSnaps):
    Scell_tmp = cp.deepcopy(Supercell)
    pos = pos0 + u_disps[isnap]
    Scell_tmp.set_positions(pos)
    Scell_snaps.append(Scell_tmp)
    snaps_ase.append(api_ph.phonopyAtoms_to_aseAtoms(Scell_tmp))

api_ph.write_Supercells_VASP(Scell_snaps,directory)


# Output xyz file

api_ph.write_xyz_aseAtomsList(snaps_ase,Snap_file)





