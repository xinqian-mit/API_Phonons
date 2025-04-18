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
import API_phonopy as api_ph
import API_phonopy_lammps as api_pl
from phonopy import Phonopy
import phonopy.file_IO as PhonIO
from phonopy.interface.calculator import get_default_physical_units
import copy as cp
import multiprocessing as mp

a = 5.65414946767
Qpoints=np.array([[0.0001,0.0001,1.0],[0.5,0.5,1.0],[3./8,3./8,3./4],[0.0,0.0,0.0],[0.5,0.5,0.5]])
band_labels=['$\Gamma$','X','K','$\Gamma$','L']

Nrepeat=[2,2,2]

prim = [[0, 0.5, 0.5],[0.5, 0, 0.5],[0.5, 0.5, 0]]
nacl = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
               cellpar=[a, a, a, 90, 90, 90])
nacl = api_ph.aseAtoms_to_phonopyAtoms(nacl)
phonon = Phonopy(nacl,np.diag(Nrepeat),primitive_matrix=prim)
phonon.generate_displacements(distance=0.03) # vasp
Scell0 = api_ph.phonopyAtoms_to_aseAtoms(phonon.supercell)

Scells_ph = phonon.supercells_with_displacements # This returns a list of Phononpy atoms object

cmds = ["pair_style eim","pair_coeff * * Na Cl ffield.eim Na Cl"]

Band_points=100
NAC = True
interface_mode = 'vasp'


forces = api_pl.calc_lmp_force_sets(cmds,Scells_ph)
phonon.forces = forces
PhonIO.write_FORCE_SETS(phonon.dataset) # write forces & displacements to FORCE_SET
force_set=PhonIO.parse_FORCE_SETS() # parse force_sets
phonon.dataset = force_set # force_set is a list of forces and displacements

if NAC == True:
    nac_params = PhonIO.get_born_parameters(
            open("BORN"),
            phonon.primitive,
            phonon.primitive_symmetry)
    if nac_params['factor'] == None:
        physical_units = get_default_physical_units(interface_mode)
        nac_params['factor'] = physical_units['nac_factor']
    phonon._nac_params=nac_params


phonon.produce_force_constants()
phonon.symmetrize_force_constants()
api_ph.write_ShengBTE_FC2(phonon.force_constants, filename='FORCE_CONSTANTS_2ND')

# calc and plot bandstructure
bands=api_ph.qpoints_Band_paths(Qpoints,Band_points)
phonon.run_band_structure(bands,with_eigenvectors=True,labels=band_labels)
phonon.write_yaml_band_structure()
bs_plt=phonon.plot_band_structure()
bs_plt.xlabel("")
bs_plt.ylabel("Frequency (THz)",fontsize=16)
bs_plt.xticks(fontsize=16)
bs_plt.yticks(fontsize=16)
bs_plt.savefig("Bandstructure.png",dpi=300,bbox_inches='tight')



