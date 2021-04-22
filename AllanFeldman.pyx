import numpy as np
from phonopy.units import VaspToTHz
from phonopy.harmonic.force_constants import similarity_transformation
#from phonopy.phonon.degeneracy import degenerate_sets

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

cpdef get_Vmat_modePair_q(double complex[:,:,:] ddm_q,double complex[:] eig_s,double complex[:] eig_r,double ws,double wr,double factor):# Dornadio's v operators. 
    
    cdef int Ns
    cdef int i
    cdef double complex[:] V_sr
    cdef double complex[:] eig_s_conj
    cdef double complex[:] eig_r_conj
    cdef double complex[:,:] ddm_q_i
    
    Ns = len(eig_s) #Natoms*3, length of the eigenvector
    eig_s_conj = np.conj(eig_s)
    V_sr = np.zeros(3,dtype='complex128')
    
    for i in range(3):
        ddm_q_i = ddm_q[i]
        V_sr[i]=np.dot(eig_s_conj,np.matmul(ddm_q_i,eig_r))*(factor**2)/2*1j/np.sqrt(np.abs(ws*wr))
       
    
    return V_sr
     


cpdef delta_lorentzian(double x,double width):
    return (width*width/4)/(x*x + width*width/4)
    
    
cpdef AF_diffusivity_q(phonon,q,double LineWidth=1e-4,double factor = VaspToTHz):
    cdef double A2m = 1e-10
    cdef double THz2Hz = 1e12
    cdef double complex[:,:,:] ddms
    cdef double complex[:,:,:] ddm_q
    cdef double complex[:,:,:] Vmat
    cdef double complex[:,:] eigvecs, 
    cdef double complex[:] SV_sr
    cdef double complex[:] SV_rs
    cdef double complex[:] V_sr
    cdef double complex[:] V_rs
    cdef double complex[:] eig_s
    cdef double complex[:] eig_r
    cdef double[:] freqs
    cdef double[:] qq
    cdef double[:] q_shifted   
    cdef double[:] Diffusivity
    cdef int s
    cdef int r
    cdef int Ns
    cdef int i
    cdef int perc
    cdef double ws
    cdef double wr
    cdef double delta

    
    qq = np.array(q)   
    
    
    dm =  phonon.get_dynamical_matrix_at_q(qq)
    eigvals, eigvecs = np.linalg.eigh(dm)
    eigvals = eigvals.real
    freqs = np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * factor  
    
    Ns = len(freqs)
    
    
    SV_sr = np.zeros(3,dtype='complex128')
    SV_rs = np.zeros(3,dtype='complex128')
    V_sr = np.zeros(3,dtype='complex128')
    V_rs = np.zeros(3,dtype='complex128')
    Vmat = np.zeros([Ns,Ns,3],dtype='complex128')
    
    
    ddms = get_dq_dynmat_q(phonon,qq)    
    if np.linalg.norm(qq) < 1e-5:
        q_shifted = np.array([1e-5,1e-5,1e-5])
        ddms = get_dq_dynmat_q(phonon,q_shifted)
    
    ddm_q = ddms[1:,:,:]
        # central derivative, need to shift by small amount to obtain the correct derivative. 
        # Otherwise will dD/dq be zero due to the fact D(q)=D(-q). 
    
    #print(ddm_q.shape)    
   
    
    
    Diffusivity = np.zeros(Ns)
    
    # compute Vmat
    for s in range(Ns):
        ws = freqs[s]*2*np.pi
        eig_s = eigvecs.T[s]
        perc = np.round(s/Ns)*10
        #print(s)
        if perc%10 < 0.001 and perc>0:
            print(str(perc)+'% completed')
        
       
        for r in range(s+1,Ns):
            wr = freqs[r]*2*np.pi
            eig_r = eigvecs.T[r]
            V_sr = get_Vmat_modePair_q(ddm_q,eig_s,eig_r,ws,wr,factor)
            #V_sr = symmetrize_gv(phonon,qq,V_sr) # symmetrize
            V_rs = -np.conj(V_sr) # antihermitians
            Vmat[s,r,:] = V_sr
            Vmat[r,s,:] = V_rs
            
            
            
            for i in range(3):            
                SV_sr[i] = V_sr[i]*(ws+wr)/2.
                SV_rs[i] = V_rs[i]*(ws+wr)/2.
             
            delta = delta_lorentzian(ws-wr,LineWidth)
            #print(lorentz)
            
            Diffusivity[s] += np.dot(np.conj(SV_sr),SV_sr).real*delta
            Diffusivity[r] += np.dot(np.conj(SV_rs),SV_rs).real*delta
                
        
        
        
    for s in range(Ns):
        ws = freqs[s]*2*np.pi
        Diffusivity[s] *= np.pi/3/(ws**2)*(A2m**2)*factor # spectral dw
        
    return Diffusivity,Vmat
            
