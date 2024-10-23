import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy import Phonopy
import phonopy.file_IO as PhonIO
from phonopy.interface.vasp import read_vasp
import API_phonopy as api_ph
import API_phonopy_lammps as api_pl
from phonopy.interface.calculator import get_default_physical_units



Prim_cell = read_vasp("POSCAR")
Ncells=[4,4,4]

# set up lammps potential, this is the same as lammps scripts
lmp_cmds = ["pair_style eim","pair_coeff * * Na Cl ffield.eim Na Cl"]

Eigfile = "NaCl-qmesh"+str(Ncells[0])+ "x" + str(Ncells[1]) +"x"+ str(Ncells[2])+"-irred.eig"
NAC=True
phonon_scell = Phonopy(Prim_cell,np.diag(Ncells)) 
phonon_scell.generate_displacements(distance=0.03) # vasp
Scells_phonopy = phonon_scell.get_supercells_with_displacements() # This returns a list of Phononpy atoms object of supercells with displacements

# convert phonopy supercells to ase atom objs.
Scells_ph = []
for scell in Scells_phonopy:
    Scells_ph.append(api_ph.phonopyAtoms_to_aseAtoms(scell))

# compute forces using lammps python interface
forces = api_pl.calc_lmp_force_sets(lmp_cmds,Scells_ph)

#parse force set and calc force constants
phonon_scell.set_forces(forces)
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



