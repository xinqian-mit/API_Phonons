import numpy as np
from phonopy.units import VaspToTHz,AMU,EV
from phonopy.harmonic.force_constants import similarity_transformation
from numba import jit,njit,prange
from API_phonopy import phonopyAtoms_to_aseAtoms,aseAtoms_to_phonopyAtoms



def get_dq_dynmat_q(phonon,q,dq=1e-5):
    #phonon._set_group_velocity()
    #_gv = phonon._group_velocity
    #ddm = _gv._get_dD(q)
    _reciprocal_lattice_inv = phonon.get_primitive().cell
    
    
    #dirx = np.dot(_reciprocal_lattice,np.array([1,0,0]))
    dq0 = dq*np.array([1,0,0])           
    dm_p0 = phonon.get_dynamical_matrix_at_q(q + dq0)
    dm_m0 = phonon.get_dynamical_matrix_at_q(q - dq0)
    ddm0 = (dm_p0 - dm_m0)/dq/2.0
    
    dq1 = dq*np.array([0,1,0])        
    dm_p1 = phonon.get_dynamical_matrix_at_q(q + dq1)
    dm_m1 = phonon.get_dynamical_matrix_at_q(q - dq1)    
    ddm1 = (dm_p1 - dm_m1)/dq/2.0
    
    dq2 = dq*np.array([0,0,1])   
    dm_p2 = phonon.get_dynamical_matrix_at_q(q + dq2)
    dm_m2 = phonon.get_dynamical_matrix_at_q(q - dq2)
    ddm2 = (dm_p2 - dm_m2)/dq/2.0    
    
    ddmx = ddm0*_reciprocal_lattice_inv[0,0] + ddm1*_reciprocal_lattice_inv[1,0] + ddm2*_reciprocal_lattice_inv[2,0]
    ddmy = ddm0*_reciprocal_lattice_inv[0,1] + ddm1*_reciprocal_lattice_inv[1,1] + ddm2*_reciprocal_lattice_inv[2,1]
    ddmz = ddm0*_reciprocal_lattice_inv[0,2] + ddm1*_reciprocal_lattice_inv[1,2] + ddm2*_reciprocal_lattice_inv[2,2]
    
    ddm = [ddmx,ddmy,ddmz]
    return ddm
    

def get_velmat_modepairs_q(phonon,q,factor=VaspToTHz):
    freqs,eigvecs = phonon.get_frequencies_with_eigenvectors(q)
    ddm = get_dq_dynmat_q(phonon,q) # three components.
    
    sqrt_fnfm = np.sqrt(freqs.T*freqs)
    
    temp_vx = np.dot(ddm[0],eigvecs)
    vx_modepairs = 1j*np.dot(eigvecs.conjugate().T,temp_vx)/sqrt_fnfm/2/np.pi*factor**2 # ATHz
    
    temp_vy = np.dot(ddm[1],eigvecs)
    vy_modepairs = 1j*np.dot(eigvecs.conjugate().T,temp_vx)/sqrt_fnfm/2/np.pi*factor**2 # ATHz
    
    temp_vz = np.dot(ddm[2],eigvecs)
    vz_modepairs = 1j*np.dot(eigvecs.conjugate().T,temp_vz)/sqrt_fnfm/2/np.pi*factor**2 # ATHz
    
    return vx_modepairs,vy_modepairs,vz_modepairs


@njit
def delta_lorentz( x, width):
    return (width)/(x*x + width*width)/np.pi


@njit 
def double_lorentz(w1,w2,width1,width2):
    return (width1+width2)/((w1-w2)*(w1-w2)+(width1+width2)**2)/np.pi

    
@njit
def delta_square(x,width):
    if np.abs(x)<width:
        return 1.0
    else:
        return 0.0
    
# Here, the linewidth is set at a fixed number. If one use the true linewidths, we get Dornadio's model.    
#@njit(parallel=True)
def calc_Diff(freqs,vx_modepairs,vy_modepairs,vz_modepairs,LineWidth=1e-2,factor=VaspToTHz):
    A2m = 1e-10
    THz2Hz = 1e12
    

    Nmodes = len(freqs) #number of modes


    Diffusivity = np.zeros(Nmodes)
            
    
    for s in range(Nmodes):
        Diff_s = 0.0
        ws = freqs[s]*2*np.pi
        for r in range(Nmodes):            
            wr = freqs[r]*2*np.pi
            wsr_avg = (ws+wr)/2.0
            tau_sr = double_lorentz(ws,wr,LineWidth,LineWidth) # THz^-1                                        
            Diff_s += wsr_avg**2*np.abs((vx_modepairs[s,r]*vx_modepairs[r,s]+vy_modepairs[s,r]*vy_modepairs[r,s]+vz_modepairs[s,r]*vz_modepairs[r,s]).real)*tau_sr/3.0       
        Diffusivity[s] = Diff_s*(A2m**2*THz2Hz)/(ws**2) #A^2THz^4/THz^2*THz-1 = A^2THz
    
    
    return Diffusivity
    


    
