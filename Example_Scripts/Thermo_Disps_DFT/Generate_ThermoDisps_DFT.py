#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Before executing this script, 
# 1. Preprocess, phonopy -d --dim="X Y Z"
# 2. After several DFT calculations, collect the vasprun.xml files
# 3. run phonopy -f ./vaspruns/*

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from phonopy import Phonopy
import phonopy.interface.vasp as Intf_vasp
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.calculator import get_default_physical_units
import phonopy.file_IO as PhonIO
import API_phonopy as api_ph # remember to set this module to python path
import phonopy.units as Units
import copy as cp
import os


# In[2]:


Qpoints=np.array([[0.,0.,0.]]) # Qpoints should always be a 2D array.
Ncells=[4,4,4] # construct supercell.
qmesh=[2,2,2] # Gamma point
Band_points=1
NSnaps = 10
Temperature = 300
directory = 'Testing_T'+str(Temperature)
Snap_file = 'Thermo_Disps_T'+str(Temperature)+'.xyz'
NAC = True
interface_mode='vasp'


# In[3]:


# generate displacements
Prim_cell = Intf_vasp.read_vasp("POSCAR") # read prim cell from the POSCAR file
Supercell = Phonopy(Prim_cell,np.diag(Ncells)).get_supercell()
Intf_vasp.write_vasp("SPOSCAR",Supercell)
phonon_scell = Phonopy(Supercell,np.eye(3,dtype=int)) #create phonon obj from the supercell.
phonon_scell.generate_displacements(distance=0.01) # vasp
Scells_phonopy = phonon_scell.get_supercells_with_displacements() # This returns a list of Phononpy atoms object


# In[4]:


force_set= PhonIO.parse_FORCE_SETS() # parse force_sets
phonon_scell.set_displacement_dataset(force_set) # force_set is a list of forces and displacements
phonon_scell.produce_force_constants()
phonon_scell.symmetrize_force_constants()


# In[5]:


if NAC == True:
    nac_params = PhonIO.get_born_parameters(
            open("BORN"),
            phonon_scell.get_primitive(),
            phonon_scell.get_primitive_symmetry())
    if nac_params['factor'] == None:
        physical_units = get_default_physical_units(interface_mode)
        nac_params['factor'] = physical_units['nac_factor']
        phonon_scell._nac_params=nac_params
    


# In[6]:


# set qpoints along BZ path
bands=api_ph.qpoints_Band_paths(Qpoints,Band_points)
phonon_scell.set_band_structure(bands, is_eigenvectors=True)
phonon_scell.set_mesh(qmesh, is_eigenvectors=True)
phonon_scell.write_yaml_band_structure()
eigvecs=api_ph.get_reshaped_eigvecs(phonon_scell)


# In[7]:


u_disps = api_ph.thermo_disp_along_eig(phonon_scell,Temperature,NSnaps)
Scell_snaps = [];
pos0 = Supercell.get_positions();


# In[8]:


for isnap in range(NSnaps):
    Scell_tmp = cp.deepcopy(Supercell)
    pos = pos0 + u_disps[isnap]
    Scell_tmp.set_positions(pos)
    Scell_snaps.append(Scell_tmp)

api_ph.write_Supercells_VASP(Scell_snaps,directory)


# In[11]:


if not os.path.exists(directory):
    os.mkdir(directory)

api_ph.write_Supercells_VASP(Scell_snaps,directory)


# Output xyz file
Scell_snaps_ase = []
for scell in Scell_snaps:
    Scell_snaps_ase.append(api_ph.phonopyAtoms_to_aseAtoms(scell))


api_ph.write_xyz_aseAtomsList(Scell_snaps_ase,Snap_file)


# In[ ]:




