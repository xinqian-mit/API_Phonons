#!/usr/bin/env python
# coding: utf-8
import matplotlib
matplotlib.use('agg')
from ase.spacegroup import crystal
from ase.visualize import view
from ase.io import write
import ase.build
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
#import API_quippy_phonopy_VASP as api_qpv
import API_phonopy as api_ph
import API_phonopy_lammps as api_pl
from phonopy import Phonopy
import phonopy.file_IO as PhonIO
from phonopy.interface.calculator import get_default_physical_units
import copy as cp
import phonopy.interface.vasp as Intf_vasp

Qpoints=[[0.0,0.0,0.0],[1./3,1./3,0.0],[0.5,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.5]]
band_labels=['$\Gamma$','K','M','$\Gamma$','A'] # remember to install texlive-latex-recommended

Nrepeat=[3,3,2]
Band_points=100
NAC = True
interface_mode = 'vasp'

ucell_ph = Intf_vasp.read_vasp("POSCAR") 
prim = np.eye(3) #[[0, 0.5, 0.5],[0.5, 0, 0.5],[0.5, 0.5, 0]]
phonon = Phonopy(ucell_ph,np.diag(Nrepeat),primitive_matrix=prim)
phonon.generate_displacements(distance=0.03) # vasp
Scell0 = api_ph.phonopyAtoms_to_aseAtoms(phonon.get_supercell())

Scells_ph = phonon.get_supercells_with_displacements() # This returns a list of Phononpy atoms object

cmds = [
"pair_style    hybrid/overlay quip ",
#"pair_style	born/coul/dsf 0.25 10.0",
"pair_coeff    * * quip ./gp_new_DB2-4.xml \"Potential xml_label=GAP_2018_8_16_-360_5_7_11_172\" 40",

#"pair_coeff	1 1 born 0.263690403 0.3170  2.340  1.04852774712 -0.499194856",
#"pair_coeff	1 2 born 0.210917632 0.3170  2.755  6.99018498080 -8.673510623",
#"pair_coeff	2 2 born 0.158231587 0.3170  3.170  72.3983444440 -145.39050181",
]




forces = api_pl.calc_lmp_force_sets(cmds,Scells_ph,'charge')
phonon.set_forces(forces)
PhonIO.write_FORCE_SETS(phonon.get_displacement_dataset()) # write forces & displacements to FORCE_SET
force_set=PhonIO.parse_FORCE_SETS() # parse force_sets
phonon.set_displacement_dataset(force_set) # force_set is a list of forces and displacements

if NAC == True:
    nac_params = PhonIO.get_born_parameters(
            open("BORN"),
            phonon.get_primitive(),
            phonon.get_primitive_symmetry())
    if nac_params['factor'] == None:
        physical_units = get_default_physical_units(interface_mode)
        nac_params['factor'] = physical_units['nac_factor']
    phonon._nac_params=nac_params


phonon.produce_force_constants()
phonon.symmetrize_force_constants()
api_ph.write_ShengBTE_FC2(phonon.get_force_constants(), filename='FORCE_CONSTANTS_2ND')

# calc and plot bandstructure
bands=api_ph.qpoints_Band_paths(Qpoints,Band_points)
phonon.set_band_structure(bands,is_eigenvectors=True,labels=band_labels)
phonon.write_yaml_band_structure()
bs_plt=phonon.plot_band_structure()
bs_plt.xlabel("")
bs_plt.ylabel("Frequency (THz)",fontsize=16)
bs_plt.xticks(fontsize=16)
bs_plt.yticks(fontsize=16)
bs_plt.savefig("Bands_lmp.png",dpi=300,bbox_inches='tight')



