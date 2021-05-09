import numpy as np
from phonopy.units import VaspToTHz,AMU,EV
from phonopy.harmonic.force_constants import similarity_transformation
from numba import jit,njit,prange


def get_dq_dynmat_q(phonon,q):
    """
    dq_scale = 1e-5 # perturbation   # Older version
    latt = phonon.primitive.cell
    Reciprocal_latt = np.linalg.inv(latt).T # recprocal lattice
        
    dm = phonon.get_dynamical_matrix_at_q(q)
    Ns,Ns1 = np.shape(dm)
    ddm_q = np.zeros([3,Ns,Ns1],dtype=np.complex128)
        
    q_abs = np.matmul(q,Reciprocal_latt) # abs coord
    for i in range(3):
        dqc = np.zeros(3)
        dqc[i] = dq_scale
        dq = np.dot(latt,dqc)
        
        qp = q + dq
        qm = q - dq
        
        dmp = phonon.get_dynamical_matrix_at_q(qp)
        dmm = phonon.get_dynamical_matrix_at_q(qm)
        
        ddm_q[i,:,:] = (dmp-dmm)/dq_scale/2.
                    
    return ddm_q
    """
    groupvel = phonon._group_velocity
    return groupvel._get_dD(q)


def get_dq_dynmat_Gamma(phonon):
    fc=phonon.get_force_constants()
    Nat,Nat1,Dim,Dim1 = np.shape(fc)
    scell = phonon.get_supercell()
    Cell_vec = phonon.get_supercell().get_cell()
    mass = scell.get_masses()
    R = phonon.get_supercell().get_positions()
    _p2s_map = phonon._dynamical_matrix._p2s_map 
    _s2p_map = phonon._dynamical_matrix._s2p_map
    multiplicity = phonon._dynamical_matrix._multiplicity
    vecs = phonon._dynamical_matrix._smallest_vectors
    
    
    dxDymat = np.zeros((Nat*3,Nat*3))
    dyDymat = np.zeros((Nat*3,Nat*3))
    dzDymat = np.zeros((Nat*3,Nat*3))
    
    for i,s_i in enumerate(_p2s_map):
        for j,s_j in enumerate(_p2s_map):
            sqrt_mm = np.sqrt(mass[i] * mass[j])
            dx_local = np.zeros((3,3))
            dy_local = np.zeros((3,3))
            dz_local = np.zeros((3,3))
            for k in range(Nat):
                if s_j == _s2p_map[k]:
                    multi = multiplicity[k][i]
                    
                    for l in range(multi):
                        vec = vecs[k][i][l] # dimensionless
                        ri_rj = np.matmul(vec,Cell_vec) # with units.
                        # Dym matrix eV/A2/AMU, [Freq]^2 
                        dx_local += fc[s_i, k] * ri_rj[0]/ sqrt_mm # eV/A/AMU
                        dy_local += fc[s_i, k] * ri_rj[1]/ sqrt_mm 
                        dz_local += fc[s_i, k] * ri_rj[2]/ sqrt_mm
                        
            dxDymat[(i*3):(i*3+3), (j*3):(j*3+3)] += dx_local
            dyDymat[(i*3):(i*3+3), (j*3):(j*3+3)] += dy_local
            dzDymat[(i*3):(i*3+3), (j*3):(j*3+3)] += dz_local
                        
                    
    ddm_dq = np.array([dxDymat,dyDymat,dzDymat])+0j             
    
    return ddm_dq
            
                                              
def symmetrize_gv(phonon,q,gv):
    symm = phonon.get_symmetry() # is an symmetry object
    
    primitive = phonon.get_primitive()
    reciprocal_lattice_inv = primitive.get_cell()
    reciprocal_lattice = np.linalg.inv(reciprocal_lattice_inv)    
    
    rotations = []
    for r in symm.get_reciprocal_operations():
        q_in_BZ = q - np.rint(q)
        diff = q_in_BZ - np.dot(r,q_in_BZ)
        if (diff < symm.get_symmetry_tolerance()).all():
            rotations.append(r)
    
    gv_sym = np.zeros_like(gv)
    
    for r in rotations:
        r_cart = similarity_transformation(reciprocal_lattice, r)
        gv_sym += np.dot(r_cart, gv.T).T
        
    return gv_sym / len(rotations)


@njit
def get_Vmat_modePair_q(ddm_q,eig_s,eig_r, ws, wr, factor):# Dornadio's v operators. 
    
    Ns = len(eig_s) #Natoms*3, length of the eigenvector
    eig_s_conj = np.conj(eig_s)
    V_sr = np.zeros(3,dtype=np.complex128)
    
    for i in range(3):
        ddm_q_i = ddm_q[i]
        V_sr[i]=np.dot(eig_s_conj,np.dot(ddm_q_i,eig_r))/2/np.sqrt(np.abs(ws*wr))
        
        #  eV/A/AMU = eV/A2/AMU * A = 2pi*factor*A
       
    V_sr = V_sr*factor**2*2*np.pi # ATHz
    
    return V_sr
     

@njit
def delta_lorentz( x, width):
    return (width)/(x*x + width*width)


@njit 
def double_lorentz(w1,w2,width1,width2):
    return (width1+width2)/((w1-w2)*(w1-w2)+(width1+width2)**2)

    
@njit
def delta_square(x,width):
    if np.abs(x)<width:
        return 1.0
    else:
        return 0.0
    
    
@njit(parallel=True)
def calc_Diff(freqs,eigvecs,ddm_q,LineWidth=1e-4,factor=VaspToTHz):
    A2m = 1e-10
    THz2Hz = 1e12
    

    Ns = len(freqs)
    Diffusivity = np.zeros(Ns)
    
    
    #SV_rs = np.zeros(3,dtype=np.complex128)
    V_sr = np.zeros(3,dtype=np.complex128)
    V_rs = np.zeros(3,dtype=np.complex128)
    Vmat = np.zeros((Ns,Ns,3),dtype=np.complex128)    
    
    # compute Vmat
    for s in prange(Ns):
 
        for r in range(s+1,Ns):
            ws = freqs[s]*2*np.pi
            eig_s = eigvecs.T[s]
            wr = freqs[r]*2*np.pi
            eig_r = eigvecs.T[r]
            V_sr = get_Vmat_modePair_q(ddm_q,eig_s,eig_r,ws,wr,factor)
            #V_sr = symmetrize_gv(phonon,q,V_sr) # symmetrize
            V_rs = get_Vmat_modePair_q(ddm_q,eig_r,eig_s,ws,wr,factor) # anti-hermitians
            Vmat[s,r,:] = V_sr
            Vmat[r,s,:] = np.conj(V_sr)
                        

    for s in prange(Ns):
        Diff_s = 0.0
        ws = freqs[s]*2*np.pi
        for r in range(Ns):
            
            wr = freqs[r]*2*np.pi
            tau_sr = delta_lorentz(ws-wr,LineWidth) # THz^-1
            #SV_sr = np.zeros(3,dtype=np.complex128)    
                                       
            Diff_s += np.dot(Vmat[s,r,:],Vmat[r,s,:]).real*tau_sr #A^2THz^2*THz-1 = A^2THz
       
        Diffusivity[s] = Diff_s*(A2m**2*THz2Hz) #A^2THz^4/THz^2*THz-1 = A^2THz
    
    
    return Diffusivity,Vmat   
    
    
    
    
def AF_diffusivity_q(phonon,q,LineWidth=1e-4,factor = VaspToTHz):

    dm =  phonon.get_dynamical_matrix_at_q(q)
    eigvals, eigvecs = np.linalg.eigh(dm)
    eigvals = eigvals.real
    freqs = np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * factor  
    
        
    if np.linalg.norm(q) < 1e-6:
        ddm_q = get_dq_dynmat_Gamma(phonon)
    else:
        ddms = get_dq_dynmat_q(phonon,q)
        ddm_q = ddms[1:,:,:]
    
    #print(ddm_q)
        # central derivative, need to shift by small amount to obtain the correct derivative. 
        # Otherwise will dD/dq be zero due to the fact D(q)=D(-q). 
      
    
    Diffusivity,Vmat = calc_Diff(freqs,eigvecs,ddm_q,LineWidth,factor) 
      
    return Diffusivity,Vmat     
