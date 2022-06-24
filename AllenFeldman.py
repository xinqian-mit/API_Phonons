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
from phonopy.phonon.degeneracy import degenerate_sets

def get_QHGK_thermal_conductvity_at_T(phonon,mesh,T,nac=False): # single temperature
    Nrepeat = phonon.get_supercell_matrix().diagonal()
    if nac:
        phono3py_cmd = 'phono3py --dim="{} {} {}" --fc2 --fc3 --nac --br --mesh="'\
                   '{} {} {}" --ts="{}"'.format(Nrepeat[0],Nrepeat[1],Nrepeat[2], 
                                                 mesh[0],mesh[1],mesh[2], str(T))   
    else:
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

        gvm = get_velmat_modepairs_q(phonon,q)
        vx_mp_q = gvm[0]
        vy_mp_q = gvm[1]
        vz_mp_q = gvm[2]
        
        #vx_mp_q,vy_mp_q,vz_mp_q = symmetrize_group_velocity_matrix_at_q(vx_mp_q,vy_mp_q,vz_mp_q,phonon,q)

        C_mp_q = calc_Cv_modepairs_q(freqs_q,T)/Vol/np.prod(mesh)*weight_q # pairwise specific heat.
        Tau_mp = Tau_modepairs_q(freqs_q,gamma_q)# in ps

        Kxxq_modes = np.real(C_mp_q*vx_mp_q.conjugate()*vx_mp_q*Tau_mp)*unit_to_WmK
        Kyyq_modes = np.real(C_mp_q*vy_mp_q.conjugate()*vy_mp_q*Tau_mp)*unit_to_WmK
        Kzzq_modes = np.real(C_mp_q*vy_mp_q.conjugate()*vy_mp_q*Tau_mp)*unit_to_WmK
        Kxyq_modes = np.real(C_mp_q*vx_mp_q.conjugate()*vy_mp_q*Tau_mp)*unit_to_WmK
        Kyzq_modes = np.real(C_mp_q*vy_mp_q.conjugate()*vz_mp_q*Tau_mp)*unit_to_WmK
        Kxzq_modes = np.real(C_mp_q*vx_mp_q.conjugate()*vz_mp_q*Tau_mp)*unit_to_WmK

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
    
    #ddm = [ddmx,ddmy,ddmz]
    return ddm[1:]


def get_velmat_modepairs_q(phonon, q, factor=VaspToTHz,cutoff_frequency=1e-4): # suitable for crystalline system.
    
    if np.linalg.norm(q) < 1e-4: # at Gamma point.
        freqs,eigvecs = phonon.get_frequencies_with_eigenvectors(q)
        ddm = get_dq_dynmat_q(phonon,q) # three components.
    
        sqrt_fnfm = np.sqrt(freqs.T*freqs)
    
        temp_vx = np.dot(ddm[0],eigvecs)*factor**2
        vx_modepairs = np.dot(eigvecs.conjugate().T,temp_vx)/sqrt_fnfm/2/(2*np.pi) # ATHz

    
        temp_vy = np.dot(ddm[1],eigvecs)*factor**2
        vy_modepairs = np.dot(eigvecs.conjugate().T,temp_vx)/sqrt_fnfm/2/(2*np.pi) # ATHz

    
        temp_vz = np.dot(ddm[2],eigvecs)*factor**2
        vz_modepairs = np.dot(eigvecs.conjugate().T,temp_vz)/sqrt_fnfm/2/(2*np.pi) # ATHz
        
        gvm = np.array([vx_modepairs,vy_modepairs,vz_modepairs])
        
        return gvm
    
    else:
    
        dm = phonon.get_dynamical_matrix_at_q(q) 
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * factor #angular
        deg_sets = degenerate_sets(freqs)
        phonon._set_group_velocity()
        _gv = phonon._group_velocity
        ddms = _gv._get_dD(q)
        rot_eigvecs = np.zeros_like(eigvecs)

        for deg in deg_sets:
            rot_eigvecs[:, deg] = rot_eigsets(ddms, eigvecs[:, deg])
        condition = freqs > cutoff_frequency
        freqs = np.where(condition, freqs, 1)
        rot_eigvecs = rot_eigvecs * np.where(condition, 1 / np.sqrt(2 * freqs), 0)

        gvm = np.zeros((3,) + eigvecs.shape,'complex')
        for i, ddm in enumerate(ddms[1:]):
            ddm = ddm * (factor**2)
            gvm[i] = np.dot(rot_eigvecs.T.conj(), np.dot(ddm, rot_eigvecs))

        if _gv._perturbation is None:
            if _gv._symmetry is None:
                return gvm
            else:
                if np.linalg.norm(q) == 0: # if at Gamma point, don't symmetrize
                    return gvm
                else:
                    return symmetrize_group_velocity_matrix(gvm, phonon, q)
        else:
            return gvm

def symmetrize_group_velocity_matrix(gvm, phonon, q):
    """Symmetrize obtained group velocity matrices.

    The following symmetries are applied:
    1. site symmetries
    2. band hermicity

    """
    # site symmetries
    _gv = phonon._group_velocity
    rotations = []
    for r in _gv._symmetry.reciprocal_operations:
        q_in_BZ = q - np.rint(q)
        diff = q_in_BZ - np.dot(r, q_in_BZ)
        if (np.abs(diff) < _gv._symmetry.tolerance).all():
            rotations.append(r)

    gvm_sym = np.zeros_like(gvm)
    for r in rotations:
        r_cart = similarity_transformation(_gv._reciprocal_lattice, r)
        gvm_sym += np.einsum("ij,jkl->ikl", r_cart, gvm)
    gvm_sym = gvm_sym / len(rotations)

    # band hermicity
    gvm_sym = (gvm_sym + gvm_sym.transpose(0, 2, 1).conj()) / 2

    return gvm_sym

def rot_eigsets( ddms, eigsets):
    """Treat degeneracy.

    Eigenvectors of degenerates bands in eigsets are rotated to make
    the velocity analytical in a specified direction (self._directions[0]).

    Parameters
    ----------
    ddms : list of ndarray
        List of delta (derivative or finite difference) of dynamical
        matrices along several q-directions for perturbation.
        shape=(len(self._directions), num_band, num_band), dtype=complex
    eigsets : ndarray
        List of phonon eigenvectors of degenerate bands.
        shape=(num_band, num_degenerates), dtype=complex

    Returns
    -------
    rot_eigvecs : ndarray
    Rotated eigenvectors.
    shape=(num_band, num_degenerates), dtype=complex

    """
    _, eigvecs = np.linalg.eigh(np.dot(eigsets.T.conj(), np.dot(ddms[0], eigsets)))
    rotated_eigsets = np.dot(eigsets, eigvecs)

    return rotated_eigsets



@njit 
def double_lorentz(w1,w2,width1,width2):
    return (width1+width2)/((w1-w2)*(w1-w2)+(width1+width2)**2)

    
@njit
def delta_square(x,width):
    if np.abs(x)<width:
        return 1.0
    else:
        return 0.0
    
# Here, the linewidth is set at a fixed number. If one use the true linewidths, we get Dornadio's model.    
#@njit(parallel=True)
def calc_Diff(freqs,gvm,LineWidth=1e-2,factor=VaspToTHz):

    vx_modepairs = gvm[0]
    vy_modepairs = gvm[1]
    vz_modepairs = gvm[2]

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
        Diffusivity[s] = Diff_s*(Angstrom**2*THz)/(ws**2) #A^2THz^4/THz^2*THz-1 = A^2THz
    
    
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
    
    Ws,Wr = np.meshgrid(freqs_THz*2*np.pi,freqs_THz*2*np.pi)
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
    Tau_sr = (np.diag(tau_s) + Tau_sr_ndiag)
    
    Tau_sr[np.isnan(Tau_sr)] = 0
    #Tau_sr[np.isnan(Tau_sr) or np.isinf(Tau_sr)] = 0
    
    return Tau_sr/2/np.pi
