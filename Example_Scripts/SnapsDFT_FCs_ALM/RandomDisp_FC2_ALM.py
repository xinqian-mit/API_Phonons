#!/usr/bin/env python3

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
from phonopy.interface.alm import get_fc2
import os
import shutil
import API_phonopy as api_ph

Qpoints=np.array([[1e-4,1e-4,1.0],[0.5,0.5,1.0],[3./8,3./8,3./4],[0.0,0.0,0.0],[0.5,0.5,0.5]])
band_labels=['$\Gamma$','X','K','$\Gamma$','L']

Ncells=[4,4,4] # Need to be consistent with the size one used to generate random displacements in DFSET

Band_points=100
NAC = True
interface_mode = 'vasp'


Unit_cell = Intf_vasp.read_vasp("POSCAR") # read prim cell from the POSCAR file
prim_mat = np.eye(3)#[[0, 0.5, 0.5],[0.5, 0, 0.5],[0.5, 0.5, 0]]
phonon = Phonopy(Unit_cell,np.diag(Ncells),primitive_matrix=prim_mat) # generate an phononpy object for LD calc.


# input random displacements
if os.path.exists('DFSET'):
    DFSET = np.loadtxt('DFSET')
    displacements=DFSET[:,0:3]
    forces=DFSET[:,3:6]
    Natoms = phonon.supercell.get_number_of_atoms()
    Nat_scells,DIM = forces.shape
    Nsnaps = int(Nat_scells/Natoms)
    forces=forces.reshape([Nsnaps,Natoms,3])
    displacements=displacements.reshape([Nsnaps,Natoms,3])

else:
    print("Cannot find DFSET!")

FC2=get_fc2(phonon.get_supercell(),phonon.get_primitive(),displacements,forces,log_level=1)

phonon.force_constants = FC2
phonon.symmetrize_force_constants()



if NAC == True:
    nac_params = PhonIO.get_born_parameters(
            open("BORN"),
            phonon.primitive,
            phonon.primitive_symmetry)
    if nac_params['factor'] == None:
        physical_units = get_default_physical_units(interface_mode)
        nac_params['factor'] = physical_units['nac_factor']
    phonon._nac_params=nac_params
    phonon._set_dynamical_matrix()



api_ph.write_ShengBTE_FC2(phonon.force_constants, filename='FORCE_CONSTANTS_2ND')
bands=api_ph.qpoints_Band_paths(Qpoints,Band_points)
phonon.run_band_structure(bands,with_eigenvectors=True,labels=band_labels)
phonon.write_yaml_band_structure()
bs_plt=phonon.plot_band_structure()
bs_plt.xlabel("")
bs_plt.ylabel("Frequency (THz)",fontsize=16)
bs_plt.xticks(fontsize=16)
bs_plt.yticks(fontsize=16)
bs_plt.savefig("Bandstructure.png",dpi=300,bbox_inches='tight')






