import numpy as np
from phonopy.units import VaspToTHz
from phonopy.harmonic.force_constants import similarity_transformation
#from phonopy.phonon.degeneracy import degenerate_sets
from numba import njit 

def get_dq_dynmat_q(phonon,q):
    """
    dq_scale = 1e-5 # perturbation   # Older version
    latt = phonon.primitive.cell
    Reciprocal_latt = np.linalg.inv(latt).T # recprocal lattice
        
    dm = phonon.get_dynamical_matrix_at_q(q)
    Ns,Ns1 = np.shape(dm)
    ddm_q = np.zeros([3,Ns,Ns1],dtype='complex128')
        
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
             
         
        
            
           
def get_Vmat_modePair_q(phonon,q,ModePair,ddms,factor = VaspToTHz): # Dornadio's v operators.
    dm =  phonon.get_dynamical_matrix_at_q(q)
    #frequencies,eigvecs = phonon.get_frequencies_with_eigenvectors(q)
    Ns,Ns1 = dm.shape
    Vmat_sr = np.zeros(3,dtype='complex128')
    #Vmat_rs = np.zeros(3,dtype='complex128')
    Vmat_sym_sr = np.zeros(3,dtype='complex128')
    #Vmat_sym_rs = np.zeros(3,dtype='complex128')
    
    eigvals, eigvecs = np.linalg.eigh(dm)
    eigvals = eigvals.real
    freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * factor  
    
    ddm_q = ddms[1:]
    
    s = ModePair[0]
    r = ModePair[1]
    
    eig = eigvecs[:,s]
    ws = freqs[s]*np.pi*2    
    
    eig1 = eigvecs[:,r]
    wr = freqs[r]*np.pi*2  
    
    for i in range(3):
        Vmat_sr[i]=np.dot(np.conj(eig),np.matmul(ddm_q[i],eig1))*(factor**2)/2/np.sqrt(np.abs(ws*wr))
        #Vmat_rs[i]=np.dot(np.conj(eig1),np.matmul(ddm_q[i],eig))*(factor**2)/2/np.sqrt(np.abs(ws*wr))
        
    Vmat_sym_sr = symmetrize_gv(phonon,q,Vmat_sr)
    #Vmat_sym_rs = symmetrize_gv(phonon,q,Vmat_rs)
    
    
    
    return Vmat_sym_sr
    

                
            

def _perturb_D(ddms, eigsets):
    """Treat degeneracy

    Group velocities are calculated using analytical continuation using
    specified directions (self._directions) in reciprocal space.

    ddms : Array-like
        List of delta (derivative or finite difference) of dynamical
            matrices along several q-directions for perturbation.
        eigsets : Array-like
            List of phonon eigenvectors of degenerate bands.

    """

    eigvals, eigvecs = np.linalg.eigh(
    np.dot(eigsets.T.conj(), np.dot(ddms[0], eigsets)))

    gv = []
    rot_eigsets = np.dot(eigsets, eigvecs)
    for ddm in ddms[1:]:
        gv.append(np.diag(np.dot(rot_eigsets.T.conj(),np.dot(ddm, rot_eigsets))).real)

    return np.transpose(gv),rot_eigsets
            
def degenerate_sets(freqs,width=1e-4,threshold=1e-4):
    indices = []
    done = []
    for i in range(len(freqs)):
        if i in done:
            continue
        else:
            f_set = [i]
            done.append(i)
        for j in range(i + 1, len(freqs)):
            lorenz_v = Lorentizan(np.abs(freqs[f_set] - freqs[j]),width)
            if ( lorenz_v < threshold).any():
            #if (np.abs(freqs[f_set] - freqs[j]) < cutoff).any():
                f_set.append(j)
                done.append(j)
        indices.append(f_set[:])

    return indices

@njit
def Lorentizan(x,width):
    return 1/np.pi*width/2/(x*x + width*width/4)
    
        
def AF_diffusivity_q(phonon,q,factor = VaspToTHz,width=1e-4,threshold=1e-4):
    A2m = 1e-10
    THz2Hz = 1e12
    ddms = get_dq_dynmat_q(phonon,q)
    freqs = phonon.get_frequencies(q)
    if np.linalg.norm(q) < 1e-5:
        q_shifted = np.array([1e-5,1e-5,1e-5])
        ddms = get_dq_dynmat_q(phonon,q_shifted)
        # central derivative, need to shift by small amount to obtain the correct derivative. 
        # Otherwise will dD/dq be zero due to the fact D(q)=D(-q). 
    #Vmat_q = get_Vmat_modes_q(phonon,q_shifted,ddms,factor) # AngstromTHz = 100m/s
    
    #SV = np.zeros_like(Vmat_q,dtype='complex128')
    
    Diffusivity = np.zeros(len(freqs))
    
    deg_sets = degenerate_sets(freqs,width,threshold) #degenerate_sets(freqs)
    
    pos = 0
    for deg in deg_sets:
        Ndeg = len(deg)
        for ideg in range(Ndeg):
            s = deg[ideg]
            
            ws = freqs[s]*2*np.pi
            for jdeg in range(Ndeg):
                r = deg[jdeg]
                wr = freqs[r]*2*np.pi
                
                #SV = Vmat_q[s,s1,:]*(ws+ws1)/2
                if s != r:
                    Vmat_sr = get_Vmat_modePair_q(phonon,q_shifted,[s,r],ddms,factor)
                    SV = Vmat_sr*(ws+wr)/2
                    
                    #print(s,s1,np.dot(np.conj(SV),SV))
                    Diffusivity[s] += np.dot(np.conj(SV),SV).real
                    #Diffusivity[s1] += np.dot(np.conj(SV),SV)
                
                
            Diffusivity[s] *= np.pi/3/ws**2 *A2m**2*THz2Hz # spectral dw
        
    return Diffusivity
    
    
    
    
    
    
    
