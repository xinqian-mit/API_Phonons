import numpy as np
from ase.spacegroup import crystal
from phonopy.structure.atoms import PhonopyAtoms
from phonopy import Phonopy
import phonopy.file_IO as PhonIO
import phonopy.interface.vasp as Intf_vasp
import API_quippy_phonopy_VASP as api_qpv
import API_phonopy_lammps as api_pl
from phonopy.interface.calculator import get_default_physical_units


cmds = ["pair_style eim","pair_coeff * * Na Cl ffield.eim Na Cl"]


a = 5.63095072418881 #5.65414946767 # lattice constant of NaCl
nacl = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
               cellpar=[a, a, a, 90, 90, 90])

Ncells=[3,3,3]
File_prefix = "NaCl-qmesh"+str(Ncells[0])+ "x" + str(Ncells[1]) +"x"+ str(Ncells[2])+"-irred"
Eigfile = File_prefix+".eig"
GropuVelFile = File_prefix+".groupVel"
NAC=True
phonon_scell = Phonopy(nacl,np.diag(Ncells),primitive_matrix=np.eye(3)) 
phonon_scell.generate_displacements(distance=0.03) # vasp
Scells_phonopy = phonon_scell.get_supercells_with_displacements() # This returns a list of Phononpy atoms object


Scells_ase = []
for scell in Scells_phonopy:
    Scells_ase.append(api_qpv.phonopyAtoms_to_aseAtoms(scell))

force_scells = api_pl.calc_lmp_force_sets(cmds,Scells_ase)

#parse force set and calc force constants
phonon_scell.set_forces(force_scells)
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
eigvecs = api_qpv.get_reshaped_eigvecs_mesh(phonon_scell)

api_qpv.write_unitcell_eigvecs_qmesh_gulp(Eigfile,eigvecs,phonon_scell)
api_qpv.write_freq_velocity_qmesh(GropuVelFile,phonon_scell)


