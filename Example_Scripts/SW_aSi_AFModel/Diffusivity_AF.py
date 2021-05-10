#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
#matplotlib.use('agg') # This line is necessary for non-gui linux systems
import matplotlib.pyplot as plt
import numpy as np
import phonopy
from phonopy import Phonopy
import phonopy.interface.vasp as Intf_vasp
from phonopy.structure.atoms import PhonopyAtoms
import phonopy.file_IO as PhonIO
from phonopy.interface.calculator import get_default_physical_units
import API_phonopy as api_ph
import AllanFeldman as AF


# In[2]:


mesh = [1,1,1]
T = 300
Gamma = np.array([0,0,0])
load_data = False

phonon = phonopy.load(supercell_matrix=[1,1,1],primitive_matrix='auto',
                     unitcell_filename="POSCAR_512_SWrelx",
                     force_constants_filename='FORCE_CONSTANTS') # load the force constants.


# In[3]:



if load_data == False:
    
    phonon.run_mesh(mesh, is_gamma_center=True, 
                with_eigenvectors=True,with_group_velocities=True,
                is_mesh_symmetry=True) # get full mesh set to false
    qpoints, weights, frequencies, eigvecs = phonon.get_mesh()
    freqs = frequencies[0]
    LineWidth = np.mean(np.diff(frequencies[0]))


    Cmodes = api_ph.mode_cv(T,freqs)

    Vol = phonon.get_supercell().get_volume()*1e-30
    
    Dmodes,Vmat = AF.AF_diffusivity_q(phonon,Gamma,LineWidth=LineWidth)
    k = np.sum(Cmodes*Dmodes/Vol)
else:
    Vol = phonon.get_supercell().get_volume()*1e-30
    freqs = np.load('freqs.npy')
    Cmodes = api_ph.mode_cv(T,freqs)
    Dmodes = np.load('Dmodes.npy')

    k = np.sum(Cmodes*Dmodes/Vol)

print(['k= '+str(k)+' W/mK'])


# In[7]:


plt.loglog(freqs,np.abs(Dmodes)*1e4,'.')
plt.xlim([1,20])
plt.ylim([1e-5,1])
plt.xlabel('Frequency (THz)')
plt.ylabel('Diffusivity ($cm^2$/s)')

np.save('Dmodes.npy',Dmodes)
np.save('freqs.npy',freqs)


# In[ ]:




