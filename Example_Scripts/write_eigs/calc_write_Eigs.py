import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy import Phonopy
import phonopy.file_IO as PhonIO
import phonopy.interface.vasp as Intf_vasp
import API_quippy_phonopy_VASP as api_qpv
from phonopy.interface.calculator import get_default_physical_units


gp_xml_file='../Dielectric_function_NaCl/soap_n12l11_6.0cut_coul/gp_NaCl_soap_coul.xml'
Prim_cell = Intf_vasp.read_vasp("POSCAR")
Ncells=[4,4,4]
Eigfile = "NaCl-qmesh"+str(Ncells[0])+ "x" + str(Ncells[1]) +"x"+ str(Ncells[2])+"-irred.eig"
NAC=True
phonon_scell = Phonopy(Prim_cell,np.diag(Ncells)) 
phonon_scell.generate_displacements(distance=0.03) # vasp
Scells_phonopy = phonon_scell.get_supercells_with_displacements() # This returns a list of Phononpy atoms object
Scells_qp = []
for scell in Scells_phonopy:
    Scells_qp.append(api_qpv.phonopyAtoms_to_aseAtoms(scell))

force_gap_scells = api_qpv.calc_force_sets_GAP(gp_xml_file,Scells_qp)

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
eigvecs = api_qpv.get_reshaped_eigvecs_mesh(phonon_scell)

api_qpv.write_unitcell_eigvecs_qmesh_gulp(Eigfile,eigvecs,phonon_scell)



