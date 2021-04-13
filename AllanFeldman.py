import numpy as np
from phonopy.units import VaspToTHz
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.phonon.degeneracy import degenerate_sets


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
             
         
        
            
            
def get_Vmat_modes_q(phonon,q,ddms,factor = VaspToTHz): # Dornadio's v operators.
    dm =  phonon.get_dynamical_matrix_at_q(q)
    #frequencies,eigvecs = phonon.get_frequencies_with_eigenvectors(q)
    Ns,Ns1 = dm.shape
    Vmat = np.zeros([Ns,Ns,3],dtype='complex128')
    Vmat_sym = np.zeros([Ns,Ns,3],dtype='complex128')
    
    eigvals, eigvecs = np.linalg.eigh(dm)
    eigvals = eigvals.real
    freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * factor
    
    #deg_sets = degenerate_sets(freqs)
    #gv = np.zeros((len(freqs), 3), dtype='double', order='C')
    
    #pos = 0 
    #eig_rot = np.zeros_like(eigvecs,dtype='complex128')
    #for deg in deg_sets: # treat degeneracy
         
    #    gv[pos:pos+len(deg)],rot_eigsets = _perturb_D(ddms, eigvecs[:, deg])
        
        
        #eig_rot[:,pos:pos+len(deg)] = rot_eigsets
    #    pos += len(deg)    
    
    ddm_q = ddms[1:]
    for s in np.arange(Ns):
        eig = eigvecs[:,s]
        fs = freqs[s]
        #Vmat[s,s,:] = gv[s]*(factor**2)/2/fs
        #Vmat_sym[s,s,:] = symmetrize_gv(phonon,q,Vmat[s,s,:])
        
        for s1 in np.arange(s,Ns):
            eig1 = eigvecs[:,s1]
            fs1 = freqs[s1]
            
            for i in range(3):
                Vmat[s,s1,i]=np.dot(np.conj(eig),np.matmul(ddm_q[i],eig1))*(factor**2)/2/np.sqrt(fs*fs1)
                Vmat[s1,s,i]=np.dot(np.conj(eig1),np.matmul(ddm_q[i],eig))*(factor**2)/2/np.sqrt(fs*fs1)
            
            Vmat_sym[s,s1,:] = symmetrize_gv(phonon,q,Vmat[s,s1,:])
            Vmat_sym[s1,s,:] = symmetrize_gv(phonon,q,Vmat[s1,s,:])
            
            #print(Vmat[s,s1,:])
            #print(Vmat_sym[s,s1,:])
                
    return Vmat_sym
                
            

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
            
        
        
        
def AF_diffusivity_q(phonon,q,factor = VaspToTHz):
    A2m = 1e-10
    THz2Hz = 1e12
    ddms = get_dq_dynmat_q(phonon,q)
    freqs = phonon.get_frequencies(q)
    Vmat_q = get_Vmat_modes_q(phonon,q,ddms,factor) # AngstromTHz = 100m/s
    
    SV = np.zeros_like(Vmat_q,dtype='complex128')
    
    Diffusivity = np.zeros(len(freqs))
    
    deg_sets = degenerate_sets(freqs)
    
    pos = 0
    for deg in deg_sets:
        Ndeg = len(deg)
        for ideg in range(Ndeg):
            s = deg[ideg]
            ws = freqs[s]*2*np.pi
            for jdeg in range(Ndeg):
                s1 = deg[jdeg]
                ws1 = freqs[s1]*2*np.pi
                
                SV = Vmat_q[s,s1,:]*(ws+ws1)/2
                if s != s1:
                    print(s,s1,np.dot(np.conj(SV),SV))
                    Diffusivity[s] += np.dot(np.conj(SV),SV).real
                    #Diffusivity[s1] += np.dot(np.conj(SV),SV)
                
                
            Diffusivity[s] *= np.pi/3/ws**2 *A2m**2*THz2Hz # spectral dw
        
    return Diffusivity
    
    
    
    
    
    
    