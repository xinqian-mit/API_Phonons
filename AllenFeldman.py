import numpy as np
from phonopy.units import VaspToTHz,AMU,EV
from phonopy.harmonic.force_constants import similarity_transformation
from numba import jit,njit,prange
from API_phonopy import phonopyAtoms_to_aseAtoms,aseAtoms_to_phonopyAtoms
import API_phonopy as api_ph
import phonopy.units as Units
from phonopy.units import EV, Angstrom, Kb, THz, THzToEv
import subprocess
import h5py

def get_QHGK_thermal_conductvity_at_T(phonon,mesh,T): # single temperature
    Nrepeat = phonon.get_supercell_matrix().diagonal()
    phono3py_cmd = 'phono3py --dim="{} {} {}" --fc2 --fc3 --br --mesh="'\
               '{} {} {}" --ts="{}"'.format(Nrepeat[0],Nrepeat[1],Nrepeat[2], 
                                             mesh[0],mesh[1],mesh[2], str(T))

    #phono3py_cmd = 'phono3py --dim="{} {} {}" --fc2 --fc3 --br --mesh="'\
    #               '{} {} {}" --ts="{}"'.format(Nrepeat[0],Nrepeat[1],Nrepeat[2], 
    #                                             mesh[0],mesh[1],mesh[2], ' '.join(str(T) for T in Temperatures))


    subprocess.call(phono3py_cmd, shell=True)
    qpoints,weights,freqs,gamma,kappaT = api_ph.read_phono3py_hdf5(mesh)
    phonon.set_mesh(mesh)
    unit_to_WmK = (Angstrom*THz)**2 /(Angstrom**3*THz)

    kxx,kyy,kzz,kxy,kyz,kxz = (0,0,0,0,0,0)
    kxx_ph,kyy_ph,kzz_ph,kxy_ph,kyz_ph,kxz_ph = (0,0,0,0,0,0)
    #CV = 0

    #K = 0

    Kxx_mp = []
    Kyy_mp = []
    Kzz_mp = []
    Vol = phonon.get_primitive().get_volume() # get volume.


    #filename = 'kappa-m{}{}{}.hdf5'.format(mesh[0],mesh[1],mesh[2])
    #ph3_data = h5py.File(filename,'r')

    #CV_qmodes = ph3_data['heat_capacity'][:][0]/Vol/np.prod(mesh)*EV

    for iq,q in enumerate(qpoints):
        #print(q)
        weight_q = weights[iq]
        freqs_q = freqs[iq] #THz
        gamma_q = gamma[0][iq] # in THz, need to convert to Trad/s.

        gamma_q[gamma_q ==0 ] = np.inf
        #tau_q = 1/(gamma_q*2)/(2*np.pi)   

        vx_mp_q,vy_mp_q,vz_mp_q = get_velmat_modepairs_q(phonon,q)

        C_mp_q = calc_Cv_modepairs_q(freqs_q,T)/Vol/np.prod(mesh)*weight_q # pairwise specific heat.
        Tau_mp = Tau_modepairs_q(freqs_q,gamma_q)# in ps

        Kxxq_modes = np.real(C_mp_q*vx_mp_q*vx_mp_q*Tau_mp)*unit_to_WmK
        Kyyq_modes = np.real(C_mp_q*vy_mp_q*vy_mp_q*Tau_mp)*unit_to_WmK
        Kzzq_modes = np.real(C_mp_q*vy_mp_q*vy_mp_q*Tau_mp)*unit_to_WmK
        Kxyq_modes = np.real(C_mp_q*vx_mp_q*vy_mp_q*Tau_mp)*unit_to_WmK
        Kyzq_modes = np.real(C_mp_q*vy_mp_q*vz_mp_q*Tau_mp)*unit_to_WmK
        Kxzq_modes = np.real(C_mp_q*vx_mp_q*vz_mp_q*Tau_mp)*unit_to_WmK

        Kxx_mp.append(Kxxq_modes)
        Kyy_mp.append(Kyyq_modes)
        Kzz_mp.append(Kzzq_modes)

        kxx += np.sum(Kxxq_modes).real
        kyy += np.sum(Kyyq_modes).real
        kzz += np.sum(Kzzq_modes).real
        kxy += np.sum(Kxyq_modes).real
        kyz += np.sum(Kyzq_modes).real
        kxz += np.sum(Kxzq_modes).real

        #gv = phonon.get_group_velocity_at_q(q)

        kxx_ph += np.trace(Kxxq_modes)
        kyy_ph += np.trace(Kyyq_modes)
        kzz_ph += np.trace(Kzzq_modes)
        kxy_ph += np.trace(Kxyq_modes)
        kyz_ph += np.trace(Kyzq_modes)
        kxz_ph += np.trace(Kxzq_modes)

    # symmetrize according to point groups.
    kappa = np.zeros((3,3))
    kappa[0,0] = kxx
    kappa[1,1] = kyy
    kappa[2,2] = kzz
    kappa[0,1] = kxy
    kappa[1,0] = kxy
    kappa[1,2] = kyz
    kappa[2,1] = kyz
    kappa[0,2] = kxz
    kappa[2,0] = kxz

    kappa_ph = np.zeros((3,3))
    kappa_ph[0,0] = kxx_ph.real
    kappa_ph[1,1] = kyy_ph.real
    kappa_ph[2,2] = kzz_ph.real
    kappa_ph[0,1] = kxy_ph.real
    kappa_ph[1,0] = kxy_ph.real
    kappa_ph[1,2] = kyz_ph.real
    kappa_ph[2,1] = kyz_ph.real
    kappa_ph[0,2] = kxz_ph.real
    kappa_ph[2,0] = kxz_ph.real

    Rot_lists = phonon.get_symmetry().get_symmetry_operations()['rotations']

    Nrots = len(Rot_lists)

    kappa_sym = np.zeros_like(kappa)
    kappa_ph_sym = np.zeros_like(kappa_ph)
    for rot in Rot_lists:
        kappa_sym += np.matmul(np.matmul(rot,kappa),np.linalg.inv(rot))
        kappa_ph_sym += np.matmul(np.matmul(rot,kappa_ph),np.linalg.inv(rot))

    kappa_sym = kappa_sym/Nrots
    kappa_ph_sym = kappa_ph_sym/Nrots
        
        
    #CV += np.trace(C_mp_q)/Angstrom**3
    return kappa_sym,kappa_ph_sym,Kxx_mp,Kyy_mp,Kzz_mp


def get_dq_dynmat_q(phonon,q,dq=1e-5):
    phonon._set_group_velocity()
    _gv = phonon._group_velocity
    ddm = _gv._get_dD(q)
    
    
    #_reciprocal_lattice_inv = phonon.get_primitive().cell    
    #dq0 = dq*np.array([1,0,0])           
    #dm_p0 = phonon.get_dynamical_matrix_at_q(q + dq0)
    #dm_m0 = phonon.get_dynamical_matrix_at_q(q - dq0)
    #ddm0 = (dm_p0 - dm_m0)/dq/2.0
    
    #dq1 = dq*np.array([0,1,0])        
    #dm_p1 = phonon.get_dynamical_matrix_at_q(q + dq1)
    #dm_m1 = phonon.get_dynamical_matrix_at_q(q - dq1)    
    #ddm1 = (dm_p1 - dm_m1)/dq/2.0
    
    #dq2 = dq*np.array([0,0,1])   
    #dm_p2 = phonon.get_dynamical_matrix_at_q(q + dq2)
    #dm_m2 = phonon.get_dynamical_matrix_at_q(q - dq2)
    #ddm2 = (dm_p2 - dm_m2)/dq/2.0    
    
    #ddmx = ddm0*_reciprocal_lattice_inv[0,0] + ddm1*_reciprocal_lattice_inv[1,0] + ddm2*_reciprocal_lattice_inv[2,0]
    #ddmy = ddm0*_reciprocal_lattice_inv[0,1] + ddm1*_reciprocal_lattice_inv[1,1] + ddm2*_reciprocal_lattice_inv[2,1]
    #ddmz = ddm0*_reciprocal_lattice_inv[0,2] + ddm1*_reciprocal_lattice_inv[1,2] + ddm2*_reciprocal_lattice_inv[2,2]
    
    #ddm = [ddmx,ddmy,ddmz]
    return ddm[1:]
    

def get_velmat_modepairs_q(phonon,q,factor=VaspToTHz):
    freqs,eigvecs = phonon.get_frequencies_with_eigenvectors(q)
    ddm = get_dq_dynmat_q(phonon,q) # three components.
    
    sqrt_fnfm = np.sqrt(freqs.T*freqs)
    
    temp_vx = np.dot(ddm[0],eigvecs)
    vx_modepairs = np.dot(eigvecs.conjugate().T,temp_vx)/sqrt_fnfm/2*factor**2 # ATHz
    #vx_modepairs = vx_modepairs.real
    #vx_modepairs = (vx_modepairs+vx_modepairs.T)/2
    
    temp_vy = np.dot(ddm[1],eigvecs)
    vy_modepairs = np.dot(eigvecs.conjugate().T,temp_vx)/sqrt_fnfm/2*factor**2 # ATHz
    #vy_modepairs = vy_modepairs.real
    #vy_modepairs = (vy_modepairs+vy_modepairs.T)/2
    
    temp_vz = np.dot(ddm[2],eigvecs)
    vz_modepairs = np.dot(eigvecs.conjugate().T,temp_vz)/sqrt_fnfm/2*factor**2 # ATHz
    #vz_modepairs = vz_modepairs.real
    #vz_modepairs = (vz_modepairs+vz_modepairs.T)/2
    
    phonon.set_group_velocity()
    gv = phonon.get_group_velocity_at_q(q)
    
    vx_diag = np.diag(vx_modepairs.diagonal())
    vx_ndiag = vx_modepairs-vx_diag
    vx_ph = np.diag(gv[:,0])
    vx_modepairs = vx_ndiag+vx_ph
    
    vy_diag = np.diag(vy_modepairs.diagonal())
    vy_ndiag = vy_modepairs-vy_diag
    vy_ph = np.diag(gv[:,1])
    vy_modepairs = vy_ndiag+vy_ph
    
    vz_diag = np.diag(vz_modepairs.diagonal())
    vz_ndiag = vz_modepairs-vz_diag
    vz_ph = np.diag(gv[:,2])
    vz_modepairs = vz_ndiag+vz_ph    
    return vx_modepairs,vy_modepairs,vz_modepairs




@njit 
def double_lorentz(w1,w2,width1,width2):
    return (width1+width2)/((w1-w2)*(w1-w2)+(width1+width2)**2)/np.pi/2.0

    
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
            if r != s:                                        
                Diff_s += wsr_avg**2*np.abs((vx_modepairs[s,r]*vx_modepairs[r,s]+vy_modepairs[s,r]*vy_modepairs[r,s]+vz_modepairs[s,r]*vz_modepairs[r,s]).real)*tau_sr/3.0       
        Diffusivity[s] = Diff_s*(A2m**2*THz2Hz)/(ws**2) #A^2THz^4/THz^2*THz-1 = A^2THz
    
    
    return Diffusivity
    

def calc_Cv_modepairs_q(freqs_THz,T):
    
    if T==0:
        n_modes = 0
        Cs = np.zeros(freqs_THz.shape)
    else:
        freqs = np.abs(freqs_THz)*THzToEv
        x = freqs / Kb / T
        expVal = np.exp(x)
        n_modes = 1/(expVal-1.0)
        Cs = Kb * x ** 2 * expVal / (expVal - 1.0) ** 2
        
    Nmodes = len(freqs_THz) #number of modes
    
    Ws,Wr = np.meshgrid(freqs_THz+1e-7,freqs_THz) # small offset
    Ns,Nr = np.meshgrid(n_modes,n_modes)
    
    Csr = Ws*Wr*(Ns-Nr)/(Wr-Ws)/T 
    Csr = Csr - np.diag(Csr.diagonal())
    Csr = Csr + np.diag(Cs) #eV/K
    Csr = Csr*EV # J/K
    
    return Csr
     


def Tau_modepairs_q(freqs_THz,gamma):

    #gamma = 2*np.pi*gamma
    Ws,Wr = np.meshgrid(2*np.pi*freqs_THz,2*np.pi*freqs_THz)
    Gs,Gr = np.meshgrid(gamma, gamma) # convert to angular frequency linewidths.
    
    Num =Gs+Gr
    Num[np.isnan(Num)] = 0
    Den = (Ws-Wr)**2+(Gs+Gr)**2
    Den[Den ==0] = np.inf
    Den[np.isnan(Den)] = np.inf
    Tau_sr = Num/Den # ps
    
    
    
    gamma[gamma == 0] =np.inf
    
    tau_s = 1/(gamma*2)
    
    Tau_sr_ndiag = Tau_sr - np.diag(Tau_sr.diagonal())
    Tau_sr = (np.diag(tau_s) + Tau_sr_ndiag)/2/np.pi
    
    Tau_sr[np.isnan(Tau_sr)] = 0
    #Tau_sr[np.isnan(Tau_sr) or np.isinf(Tau_sr)] = 0
    
    return Tau_sr
