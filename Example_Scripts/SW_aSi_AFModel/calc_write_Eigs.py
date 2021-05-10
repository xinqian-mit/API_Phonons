import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy import Phonopy
import phonopy.file_IO as PhonIO
import phonopy.interface.vasp as Intf_vasp
import API_phonopy as api_ph
import API_quippy as api_q
from phonopy.interface.calculator import get_default_physical_units
from ase.calculators.lammpslib import LAMMPSlib


Prim_cell = Intf_vasp.read_vasp("POSCAR_512_SWrelx")

pot_file ='Si.sw'
lammps_header = [
    'units metal',
    'atom_style atomic',
    'atom_modify map array sort 0 0',
    #'box tilt large',
]

cmds = [
    "pair_style sw",
    "pair_coeff * * Si.sw Si",
]

Ncells=[1,1,1]
Eigfile = "aSi-qmesh"+str(Ncells[0])+ "x" + str(Ncells[1]) +"x"+ str(Ncells[2])+"-sw-irred.eig"
NAC=False
phonon_scell = Phonopy(Prim_cell,np.diag(Ncells)) 
phonon_scell.generate_displacements(distance=0.03) # vasp
Scells_phonopy = phonon_scell.get_supercells_with_displacements() # This returns a list of Phononpy atoms object

force_gap_scells = []
for scell in Scells_phonopy:
    scell_ase = api_ph.phonopyAtoms_to_aseAtoms(scell)
    lmp = LAMMPSlib(lmpcmds=cmds, log_file='log.lammps',lammps_header=lammps_header)
    scell_ase.set_calculator(lmp)
    force = scell_ase.get_forces().tolist()
    force_gap_scells.append(force)

#parse force set and calc force constants
phonon_scell.set_forces(force_gap_scells)
PhonIO.write_FORCE_SETS(phonon_scell.get_displacement_dataset()) # write forces & displacements to FORCE_SET
force_set=PhonIO.parse_FORCE_SETS() # parse force_sets
phonon_scell.set_displacement_dataset(force_set) # force_set is a list of forces and displacements

if NAC == True:
    nac_params = PhonIO.get_born_parameters(
        open("BORN"),
        phonon_scell.get_primitive(),
        phonon_scell.get_primitive_symmetry())
    if nac_params['factor'] == None:
        physical_units = get_default_physical_units('vasp')
        nac_params['factor'] = physical_units['nac_factor']
    phonon_scell._nac_params=nac_params

phonon_scell.produce_force_constants()
phonon_scell.symmetrize_force_constants()
PhonIO.write_FORCE_CONSTANTS(phonon_scell.get_force_constants(), filename='FORCE_CONSTANTS')


phonon_scell.set_mesh(Ncells, is_gamma_center=True, is_eigenvectors=True)
eigvecs = api_ph.get_reshaped_eigvecs_mesh(phonon_scell)

api_ph.write_unitcell_eigvecs_qmesh_gulp(Eigfile,eigvecs,phonon_scell)



