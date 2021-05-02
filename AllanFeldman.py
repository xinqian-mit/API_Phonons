import numpy as np
from phonopy.units import VaspToTHz
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
                        
                        dx_local += fc[s_i, k] * ri_rj[0]/ sqrt_mm 
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
        V_sr[i]=np.dot(eig_s_conj,np.dot(ddm_q_i,eig_r))*(factor**2)/2*1j/np.sqrt(np.abs(ws*wr))
       
    
    return V_sr
     

@njit
def delta_lorentz( x, width):
    return (width/2)/(x*x + width*width/4)/np.pi
    
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
    Diff_s = 0.0


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
            V_rs = -np.conj(V_sr) # antihermitians
            #print(V_sr)
            Vmat[s,r,:] = V_sr
            Vmat[r,s,:] = V_rs
                        

    for s in prange(Ns):
        Diff_s = 0.0
        ws = freqs[s]*2*np.pi
        for r in range(Ns):
            
            wr = freqs[r]*2*np.pi
            delta = delta_lorentz(ws-wr,LineWidth)
            SV_sr = np.zeros(3,dtype=np.complex128)    
            for i in range(3):            
                SV_sr[i] = Vmat[s,r,i]*(ws+wr)/2.
                            
            Diff_s += np.dot(np.conj(SV_sr),SV_sr).real*delta
       
        Diffusivity[s] = Diff_s*np.pi/3/(ws**2)*(A2m**2*THz2Hz)
    
    
    return Diffusivity,Vmat   
    
    
    
    
def AF_diffusivity_q(phonon,q,LineWidth=1e-4,factor = VaspToTHz):

    
    
    dm =  phonon.get_dynamical_matrix_at_q(q)
    eigvals, eigvecs = np.linalg.eigh(dm)
    eigvals = eigvals.real
    freqs = np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * factor  
    
    
    

    
        
    if np.linalg.norm(q) < 1e-5:
        ddm_q = get_dq_dynmat_Gamma(phonon)
    else:
        ddms = get_dq_dynmat_q(phonon,q)
        ddm_q = ddms[1:,:,:]
    
    
    
    
    
    #print(ddm_q)
        # central derivative, need to shift by small amount to obtain the correct derivative. 
        # Otherwise will dD/dq be zero due to the fact D(q)=D(-q). 
    
    #print(ddm_q.shape)   
    
    Diffusivity,Vmat = calc_Diff(freqs,eigvecs,ddm_q,LineWidth,factor) 
   
    #print(Vmat)
    

        
    return Diffusivity,Vmat     
