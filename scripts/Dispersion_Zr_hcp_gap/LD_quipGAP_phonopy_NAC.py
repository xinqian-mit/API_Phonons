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
import API_quippy_phonopy_VASP as api_qpv # remember to set this module to python path


gp_xml_file="gp_new_DB2-4.xml"
Qpoints=[[0.0,0.0,0.0],[1./3,1./3,0.0],[0.5,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.5]]
band_labels=['$\Gamma$','K','M','$\Gamma$','A'] # remember to install texlive-latex-recommended
Ncells=[3,3,2]
Band_points=50
NAC = False
interface_mode = 'vasp'


# generate displacements
Unit_cell = Intf_vasp.read_vasp("POSCAR") # read prim cell from the POSCAR file
prim_mat = np.eye(3)#[[0, 0.5, 0.5],[0.5, 0, 0.5],[0.5, 0.5, 0]]
phonon_scell = Phonopy(Unit_cell,np.diag(Ncells),primitive_matrix=prim_mat) # generate an phononpy object for LD calc.
phonon_scell.generate_displacements(distance=0.03) # vasp
Scells_phonopy = phonon_scell.get_supercells_with_displacements() # This returns a list of Phononpy atoms object

# convert phonopy atoms objects to quippy atom objects
Scells_quippy =[]
for scell in Scells_phonopy:
    Scells_i=api_qpv.phonopyAtoms_to_aseAtoms(scell) 
    Scells_quippy.append(Scells_i)

# calculate forces and convert to phonopy force_sets
force_gap_scells = api_qpv.calc_force_sets_GAP(gp_xml_file,Scells_quippy)

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
        physical_units = get_default_physical_units(interface_mode)
        nac_params['factor'] = physical_units['nac_factor']
    phonon_scell._nac_params=nac_params


phonon_scell.produce_force_constants()
phonon_scell.symmetrize_force_constants()
api_qpv.write_ShengBTE_FC2(phonon_scell.get_force_constants(), filename='FORCE_CONSTANTS_2ND')

# calc and plot bandstructure
bands=api_qpv.qpoints_Band_paths(Qpoints,Band_points)
phonon_scell.set_band_structure(bands,labels=band_labels)
phonon_scell.write_yaml_band_structure()
bs_plt=phonon_scell.plot_band_structure()
bs_plt.xlabel("")
bs_plt.ylabel("Frequency (THz)",fontsize=16)
bs_plt.xticks(fontsize=16)
bs_plt.yticks(fontsize=16)
bs_plt.savefig("Bandstructure.png",dpi=300,bbox_inches='tight')


