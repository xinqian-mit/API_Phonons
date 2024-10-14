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
import API_quippy as api_q # remember to set this module to python path
import API_phonopy as api_ph

gp_xml_file="gp_new_DB2-4.xml"
Qpoints=[[0.0,0.0,0.0],[1./3,1./3,0.0],[0.5,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.5]]
band_labels=['$\Gamma$','K','M','$\Gamma$','A'] # remember to install texlive-latex-recommended
Ncells=[3,3,2]
Band_points=50
NAC = False
interface_mode = 'vasp'


Relax = True

if Relax: 
    from ase.optimize import BFGS
    import quippy

    fmax = 1e-6
    pot = quippy.potential.Potential('IP GAP',param_filename=gp_xml_file)
    
    # you might need to comment self.name = args_str in  potential.py of the quippy library

    ucell = Intf_vasp.read_vasp('POSCAR')
    ucell_ase = api_ph.phonopyAtoms_to_aseAtoms(ucell)
    ucell_ase.calc = pot # ase can accept quippy potential object as calculators
    #ucell_ase.set_constraint(FixAtoms(mask=[True for atom in ucell_ase]))
    #ucf = UnitCellFilter(ucell_ase)
    relax = BFGS(ucell_ase)

    relax.run(fmax=fmax)
    Unit_cell = api_ph.aseAtoms_to_phonopyAtoms(ucell_ase)
    Intf_vasp.write_vasp('POSCAR_relx',Unit_cell)
    
else:
    Unit_cell = Intf_vasp.read_vasp("POSCAR") # read prim cell from the POSCAR file


# generate displacements

prim_mat = np.eye(3)#[[0, 0.5, 0.5],[0.5, 0, 0.5],[0.5, 0.5, 0]]
phonon_scell = Phonopy(Unit_cell,np.diag(Ncells),primitive_matrix=prim_mat) # generate an phononpy object for LD calc.
phonon_scell.generate_displacements(distance=0.03) # vasp
Scells_phonopy = phonon_scell.get_supercells_with_displacements() # This returns a list of Phononpy atoms object

# convert phonopy atoms objects to quippy atom objects
Scells_quippy = []
for scell in Scells_phonopy:
    scell_qp = api_ph.phonopyAtoms_to_aseAtoms(scell)
    Scells_quippy.append(scell_qp)
 

# calculate forces and convert to phonopy force_sets
force_gap_scells = api_q.calc_force_sets_GAP(gp_xml_file,Scells_quippy)

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
api_ph.write_ShengBTE_FC2(phonon_scell.get_force_constants(), filename='FORCE_CONSTANTS_2ND')
# phonopy 2.7 changed format, ShengBTE won't read, use the file in api_qpv to write.

# calc and plot bandstructure
bands=api_ph.qpoints_Band_paths(Qpoints,Band_points)
phonon_scell.set_band_structure(bands,is_eigenvectors=True,labels=band_labels)
phonon_scell.write_yaml_band_structure()
bs_plt=phonon_scell.plot_band_structure()
bs_plt.xlabel("")
bs_plt.ylabel("Frequency (THz)",fontsize=16)
bs_plt.xticks(fontsize=16)
bs_plt.yticks(fontsize=16)
bs_plt.savefig("Bandstructure.png",dpi=300,bbox_inches='tight')

