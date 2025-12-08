import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.units import EV, Angstrom, THz, Hbar, THzToEv
from phonopy.phonon.thermal_properties import mode_cv
import h5py
from scipy.interpolate import RegularGridInterpolator

#import multiprocessing as mp




def Generate_TTGMeshGrid(Grating_Period,FreqH_MHz,sparse=True):

    q_TTG = 2*np.pi/Grating_Period
    Xix = np.array([-q_TTG,q_TTG])
    OmegaH_Trads = 2*np.pi*FreqH_MHz*1e-6
        
    return np.meshgrid(Xix,OmegaH_Trads,sparse=sparse , indexing='ij')


def Generate_1DMeshGrid(Scale_cutoff_r,Scale_meshsize_r,kappa_r,Cap,FreqH_MHz,rp,rs,sparse=True):

    dpr_max = np.sqrt(kappa_r/np.pi/Cap/np.min(1e6*np.abs(FreqH_MHz)))
    dpr_min = np.sqrt(kappa_r/np.pi/Cap/np.max(1e6*np.abs(FreqH_MHz)))
    
    OmegaH_Trads = 2*np.pi*FreqH_MHz*1e-6
    rpp = np.sqrt(rp*rp+rs*rs)


    inv_Rcutoff = Scale_cutoff_r/np.mean([rpp,dpr_min])
    inv_Rmeshsize = np.min([4*np.sqrt(2)/rpp,4/dpr_max])/Scale_meshsize_r


    # now, generate linear mesh.
    Xir = np.arange(0,inv_Rcutoff,inv_Rmeshsize) 


    return np.meshgrid(Xir,OmegaH_Trads,sparse=sparse , indexing='ij')


def Generate_2Dxy_MeshGrid(Scale_cutoffs_xy,Scale_meshsizes_xy,kappa_x,kappa_y,Cap,FreqH_MHz,rp,rs,R0=None,sparse=True):

    eps = 1e-20
    dpx_max = np.sqrt(kappa_x/np.pi/Cap/np.min(1e6*np.abs(FreqH_MHz)))
    dpx_min = np.sqrt(kappa_x/np.pi/Cap/np.max(1e6*np.abs(FreqH_MHz)))
    dpy_max = np.sqrt(kappa_y/np.pi/Cap/np.min(1e6*np.abs(FreqH_MHz)))
    dpy_min = np.sqrt(kappa_y/np.pi/Cap/np.max(1e6*np.abs(FreqH_MHz)))
    OmegaH_Trads = 2*np.pi*FreqH_MHz*1e-6
    rpp = np.sqrt(rp*rp+rs*rs)

    Scale_cutoff_x = Scale_cutoffs_xy[0]
    Scale_cutoff_y = Scale_cutoffs_xy[1]
    Scale_meshsize_x = Scale_meshsizes_xy[0]
    Scale_meshsize_y = Scale_meshsizes_xy[1]
    
    if R0 == None:
        R0 = np.mean(rp)
    
    if type (rpp) == int or type (rpp) == float or type (rpp) == np.float64:
        inv_Xcutoff = Scale_cutoff_x/np.mean([rpp,dpx_min])
        inv_Ycutoff = Scale_cutoff_x/np.mean([rpp,dpy_min])
        
        inv_Xmeshsize = np.min([2*np.sqrt(2)/rpp,2/dpx_max])/Scale_meshsize_x
        inv_Ymeshsize = np.min([2*np.sqrt(2)/rpp,2/dpy_max])/Scale_meshsize_y
    
    elif type(rpp)== tuple or type(rpp)== list or type (rpp) == np.ndarray: 
        inv_Xcutoff = Scale_cutoff_x/np.mean([rpp[0],dpx_min])
        inv_Ycutoff = Scale_cutoff_y/np.mean([rpp[1],dpy_min])
        
        inv_Xmeshsize = np.min([2*np.sqrt(2)/rpp[0],2/dpx_max])/Scale_meshsize_x
        inv_Ymeshsize = np.min([2*np.sqrt(2)/rpp[1],2/dpy_max])/Scale_meshsize_y
        
    else:
        inv_Xcutoff = Scale_cutoff_x/np.mean(rpp)
        inv_Ycutoff = Scale_cutoff_x/np.mean(rpp)
        
        inv_Xmeshsize = 2*np.sqrt(2)/np.mean(rpp)/Scale_meshsize_x
        inv_Ymeshsize = 2*np.sqrt(2)/np.mean(rpp)/Scale_meshsize_y        
        

    # now, generate linear mesh.
    
    
    Xix =  np.arange(eps,inv_Xcutoff,inv_Xmeshsize)
    Xiy =  np.arange(eps,inv_Ycutoff,inv_Ymeshsize)

    return np.meshgrid(Xix,Xiy,OmegaH_Trads,sparse=sparse , indexing='ij')


def Generate_2Drz_MeshGrid(Scale_cutoffs_rz,Scale_meshsizes_rz,kappa_r,kappa_z,Cap,FreqH_MHz,rp,rs,skin_depths=None,sparse=True):

    dpr_max = np.sqrt(kappa_r/np.pi/Cap/np.min(1e6*np.abs(FreqH_MHz)))
    dpr_min = np.sqrt(kappa_r/np.pi/Cap/np.max(1e6*np.abs(FreqH_MHz)))
    dpz_max = np.sqrt(kappa_z/np.pi/Cap/np.min(1e6*np.abs(FreqH_MHz)))
    dpz_min = np.sqrt(kappa_z/np.pi/Cap/np.max(1e6*np.abs(FreqH_MHz)))
    OmegaH_Trads = 2*np.pi*FreqH_MHz*1e-6
    rpp = np.sqrt(rp*rp+rs*rs)

    Scale_cutoff_r = Scale_cutoffs_rz[0]
    Scale_cutoff_z = Scale_cutoffs_rz[1]
    Scale_meshsize_r = Scale_meshsizes_rz[0]
    Scale_meshsize_z = Scale_meshsizes_rz[1]
    inv_Rcutoff = Scale_cutoff_r/np.mean([rpp,dpr_min])
    inv_Rmeshsize = np.min([4*np.sqrt(2)/rpp,4/dpr_max])/Scale_meshsize_r
    
    if skin_depths == None:
        inv_zcutoff = Scale_cutoff_r/np.mean([rpp,dpz_min])
        inv_zmeshsize = np.min([4*np.sqrt(2)/rpp,4/dpz_max])/Scale_meshsize_r
        
    else:
        inv_zcutoff = np.max([Scale_cutoff_z/dpz_min,1/np.mean(skin_depths)])
        inv_zmeshsize = np.min([1/dpz_min/Scale_meshsize_z,1/np.max(skin_depths)/Scale_meshsize_z])

    # now, generate linear mesh.
    Xir = np.arange(0,inv_Rcutoff,inv_Rmeshsize) 
    Xiz = np.arange(0,inv_zcutoff,inv_zmeshsize)

    return np.meshgrid(Xir,Xiz,OmegaH_Trads,sparse=sparse , indexing='ij')

def Generate_xyzMeshGrid(Scale_cutoffs_xyz,Scale_meshsizes_xyz,kappa_x,kappa_y,kappa_z,Cap,FreqH_MHz,rp,rs,skin_depths,sparse=True):

    dpx_max = np.sqrt(kappa_x/np.pi/Cap/np.min(1e6*np.abs(FreqH_MHz)))
    dpx_min = np.sqrt(kappa_x/np.pi/Cap/np.max(1e6*np.abs(FreqH_MHz)))
    dpy_max = np.sqrt(kappa_y/np.pi/Cap/np.min(1e6*np.abs(FreqH_MHz)))
    dpy_min = np.sqrt(kappa_y/np.pi/Cap/np.max(1e6*np.abs(FreqH_MHz)))
    
    dpz_min = np.sqrt(kappa_z/np.pi/Cap/np.max(1e6*np.abs(FreqH_MHz)))
    OmegaH_Trads = 2*np.pi*FreqH_MHz*1e-6
    
    if type(rp) == float or type(rp) == int or type(rp)==np.float64:
        rppx = np.sqrt(rp**2+rs**2)
        rppy = rppx
    
    if type(rp) == tuple or type(rp) == tuple or type(rp) ==  np.ndarray:
        rppx = np.sqrt(rp[0]*rp[0]+rs[0]*rs[0])
        rppy = np.sqrt(rp[1]*rp[1]+rs[1]*rs[1])
        

    Scale_cutoff_x = Scale_cutoffs_xyz[0]
    Scale_cutoff_y = Scale_cutoffs_xyz[1]
    Scale_cutoff_z = Scale_cutoffs_xyz[2]
    
    Scale_meshsize_x = Scale_meshsizes_xyz[0]
    Scale_meshsize_y = Scale_meshsizes_xyz[1]
    Scale_meshsize_z = Scale_meshsizes_xyz[2]
    
    inv_xcutoff = Scale_cutoff_x/np.mean([rppx,dpx_min])
    inv_xmeshsize = np.min([4*np.sqrt(2)/rppx,4/dpx_max])/Scale_meshsize_x
    inv_ycutoff = Scale_cutoff_y/np.mean([rppy,dpy_min])
    inv_ymeshsize = np.min([4*np.sqrt(2)/rppy,4/dpy_max])/Scale_meshsize_y
    inv_zcutoff = np.max([Scale_cutoff_z/dpz_min,1/np.mean(skin_depths)])
    inv_zmeshsize = np.min([1/dpz_min/Scale_meshsize_z,1/np.max(skin_depths)/Scale_meshsize_z])

    # now, generate linear mesh.
    Xix = np.arange(0,inv_xcutoff,inv_xmeshsize) 
    Xiy = np.arange(0,inv_ycutoff,inv_ymeshsize)
    Xiz = np.arange(0,inv_zcutoff,inv_zmeshsize)

    return np.meshgrid(Xix,Xiy,Xiz,OmegaH_Trads,sparse=sparse , indexing='ij')
    
def Interp_InvSpatio_GF(Meshgrid_in,GreenFunc,Scale_spmesh,sparse=True,method='linear'):
    sptDim = len(Meshgrid_in) # frequency domain is the last.
    if sptDim == 2:
        XI,OMEGAH = Meshgrid_in
        Xi = XI[:,0]
        OmegaH = OMEGAH[0,:]
        
        Scale_r = Scale_spmesh
        Xif = np.linspace(0,np.max(XI),int(Scale_r*len(Xi)))
        
        Reinterp = RegularGridInterpolator((Xi, OmegaH),np.real(GreenFunc),method=method)
        Iminterp = RegularGridInterpolator((Xi, OmegaH),np.imag(GreenFunc),method=method)
        
        XIf,OMEGAH = np.meshgrid(Xif,OmegaH,indexing='ij',sparse=sparse)
        
        Re_GF_interp = Reinterp((XIf,OMEGAH))
        Im_GF_interp = Iminterp((XIf,OMEGAH))
        
        GF_interp = Re_GF_interp+1j*Im_GF_interp
        
        return (XIf,OMEGAH),GF_interp
    elif sptDim == 3:
        XIr,XIz,OMEGAH = Meshgrid_in
        Scale_r = Scale_spmesh[0]
        Scale_z = Scale_spmesh[1]
        
        Xir = XIr[:,0,0]
        Xiz = XIz[0,:,0]
        OmegaH = OMEGAH[0,0,:]
        
        Xirf = np.linspace(0,np.max(XIr),int(Scale_r*len(Xir)))
        Xizf = np.linspace(0,np.max(XIz),int(Scale_z*len(Xiz)))
        
        Reinterp = RegularGridInterpolator((Xir, Xiz, OmegaH),np.real(GreenFunc),method=method)
        Iminterp = RegularGridInterpolator((Xir, Xiz, OmegaH),np.imag(GreenFunc),method=method)
        XIrf,XIzf,OMEGAH = np.meshgrid(Xirf,Xizf,OmegaH,indexing='ij',sparse=sparse)
        
        Re_GF_interp = Reinterp((XIrf,XIzf,OMEGAH))
        Im_GF_interp = Iminterp((XIrf,XIzf,OMEGAH))
        
        GF_interp = Re_GF_interp+1j*Im_GF_interp
        
        return (XIrf,XIzf,OMEGAH),GF_interp
    elif sptDim == 4:
        XIx,XIy,XIz,OMEGAH = Meshgrid_in
        Scale_x = Scale_spmesh[0]
        Scale_y = Scale_spmesh[1]
        Scale_z = Scale_spmesh[2]
        
        Xix = XIx[:,0,0,0]
        Xiy = XIy[0,:,0,0]
        Xiz = XIz[0,0,:,0]
        
        Xixf = np.linspace(0,np.max(XIx),int(Scale_x*len(Xix)))
        Xiyf = np.linspace(0,np.max(XIx),int(Scale_y*len(Xiy)))
        Xizf = np.linspace(0,np.max(XIz),int(Scale_z*len(Xiz)))
        OmegaH = OMEGAH[0,0,0,:] # the time-domain will be interpolated later.
        
        Reinterp = RegularGridInterpolator((Xix,Xiy,Xiz,OmegaH),np.real(GreenFunc),method=method)
        Iminterp = RegularGridInterpolator((Xix,Xiy,Xiz,OmegaH),np.imag(GreenFunc),method=method)
        
        XIxf,XIyf,XIzf,OMEGAH = np.meshgrid(Xixf,Xiyf,Xizf,OmegaH,indexing='ij',sparse=sparse)
        Re_GF_interp = Reinterp((XIxf,XIyf,XIzf,OMEGAH))
        Im_GF_interp = Iminterp((XIxf,XIyf,XIzf,OMEGAH))
        GF_interp = Re_GF_interp+1j*Im_GF_interp
        
        return (XIxf,XIyf,XIzf,OMEGAH),GF_interp
    else:
        return None
    
# read vectorial phonon properties from ShengBTE outputs, such as gv and MFD.
def import_BTE_PhvecProps(file,Ns):
    #Ns = phonon.get_unitcell().get_number_of_atoms()*3    
    vec_modes_in = np.loadtxt(file)
    
    Nmodes = len(vec_modes_in)
    Nq = int(Nmodes/Ns)
    
    vecx_modes = np.reshape(vec_modes_in[:,0],(Ns,Nq)).T
    vecy_modes = np.reshape(vec_modes_in[:,1],(Ns,Nq)).T
    vecz_modes = np.reshape(vec_modes_in[:,2],(Ns,Nq)).T
    
    vec_modes = np.array([vecx_modes,vecy_modes,vecz_modes])
    return vec_modes

# Flatten the indices q&s to a single index
def Vectorize_mode_props(Mode_props):
    (Nq,Ns)=Mode_props.shape;
    return np.reshape(Mode_props,Nq*Ns)

def expand_qired2qfull(ph_prop,qpoints_full_list):
    # expand scalor modal properties at irreducible qpoints to full mesh
    (Nqired,Ns) = ph_prop.shape
    Nq = len(qpoints_full_list)
    qfull2irred = qpoints_full_list[:,:2]
    qfull2irred = qfull2irred.astype(int)-1 # convert to python index starting from 0
    prop_full = np.zeros((Nq,Ns))
    for iq in range(Nq):
        iq_ired = qfull2irred[iq,1]
        for s in range(Ns):
            prop_full[iq,s] = ph_prop[iq_ired,s]
            
            
    return prop_full

# map the MFD at full mesh to the irreducible wedge.
def vFmodes_full_to_irep(F_modes_full,D_boundary,gv_full,gv_ired,tau_ired,freqs_ired,qpoints_full_list,tau_Dims=[0,1,2]):
    
    eps = 1e-50
    
    if type(tau_Dims) is tuple:
        tau_Dims = list(tau_Dims)
    
    Dim,Nqf,Ns = gv_full.shape
    Dim,Nqired,Ns = gv_ired.shape
    Nmodes_ired = Nqired*Ns

    qfull_to_irred = qpoints_full_list[:,:2].astype(int)-1
    #qpoints_full = qpoints_full_list[:,2:]
    

    vF_modes_ired = np.zeros((3,3,Nmodes_ired))
    F_modes_ired = np.zeros((3,Nmodes_ired))
    vx_modes_ired = Vectorize_mode_props(gv_ired[0])
    vy_modes_ired = Vectorize_mode_props(gv_ired[1])
    vz_modes_ired = Vectorize_mode_props(gv_ired[2])
    gv_modes_ired = np.array([vx_modes_ired,vy_modes_ired,vz_modes_ired])
    
    Vec_tau_ired = Vectorize_mode_props(tau_ired)
    
    Vec_tau_sc = np.zeros(Nmodes_ired)
    
    weights = np.zeros(Nqired)
    
    for iqfull in range(Nqf):
        iq = qfull_to_irred[iqfull,1]
        weights[iq] +=1 

    for iqfull in range(Nqf):
        iq = qfull_to_irred[iqfull,1] # map to irreducible qpoint
        for s in range(Ns):
            iqs = iq*Ns+s
            Fqfs =  F_modes_full[:,iqfull,s]/(freqs_ired[iq,s]+eps)/2/np.pi*10 # Angstrom 
            Fqfs = Fqfs/(1+2*np.abs(Fqfs)/(D_boundary/Angstrom))
            vqfs =  gv_full[:,iqfull,s] # Angstrom THz
            
            vF_modes_ired[:,:,iqs] += np.tensordot(vqfs,Fqfs,axes=0)/weights[iq]
            
            Num = 0
            Den = 0
     
            for ix in tau_Dims:
                
                Num += vqfs[ix]*Fqfs[ix]
                Den += vqfs[ix]*vqfs[ix]
                
            Vec_tau_sc[iqs] += Num/Den/weights[iq] # rescaled relaxatio time. 
            
            
            
            
   
    Vec_tau_sc[np.isnan(Vec_tau_sc)] = Vec_tau_ired[np.isnan(Vec_tau_sc)]# 0.0
    Vec_tau_sc[np.abs(Vec_tau_sc)>np.sqrt(THz)] = Vec_tau_ired[np.abs(Vec_tau_sc)>np.sqrt(THz)]
                        
    for i in range(Dim):
       MFP2_i = vF_modes_ired[i,i,:]*Vec_tau_sc
       F_modes_ired[i] = np.sqrt(np.abs(MFP2_i))*np.sign(MFP2_i)*np.sign(gv_modes_ired[i])
    

    return vF_modes_ired,F_modes_ired,np.abs(Vec_tau_sc)




def load_ShengBTE_KCM(T0,phonon,Dir_BTE_HarPhons,Dir_BTE_MFD,Dir_BTE_lifetime,D_boundary,is_isotope,four_phonon=False,calc_fullmesh=False,eta_Dims=[0,1],Gamma_cut=False):
    # Loading
    eps = 1e-50
    omega = np.loadtxt(Dir_BTE_HarPhons+'BTE.omega') # 2*pi*Thz
    qpoints_list = np.loadtxt(Dir_BTE_HarPhons+'BTE.qpoints')
    qpoints_full_list = np.loadtxt(Dir_BTE_HarPhons+'BTE.qpoints_full')
    try:
        RecLatt = np.loadtxt(Dir_BTE_HarPhons+'BTE.ReciprocalLatticeVectors') # nm-1
    except:
        RecLatt = np.linalg.inv( phonon.get_unitcell().get_cell())*2*np.pi*10 #
    v_full = np.loadtxt(Dir_BTE_HarPhons+'BTE.v_full') # km/s
    w_ph = np.loadtxt(Dir_BTE_lifetime+'BTE.w_3ph')[:,-1] # THz
    w_ph_N = np.loadtxt(Dir_BTE_lifetime+'BTE.w_3ph_Normal')[:,-1] # THz
    F_final = np.loadtxt(Dir_BTE_MFD+'BTE.F_final') # nmTHz
    if four_phonon:
        w_ph += np.loadtxt(Dir_BTE_lifetime+'BTE.w_4ph')[:,-1]
        w_ph_N += np.loadtxt(Dir_BTE_lifetime+'BTE.w_4ph_Normal')[:,-1] 
    if is_isotope:
        w_ph += np.loadtxt(Dir_BTE_HarPhons+'BTE.w_isotopic')[:,1]
        
    # Phonon qpoints
    ired = qpoints_list[:,1].astype(int)-1
    full = qpoints_full_list[:,1].astype(int)-1
    Nq,(Nqired,Ns) = len(full),omega.shape
    qpoints_full = qpoints_full_list[:,2:]
    mesh = np.round(1/(1-qpoints_full[-1])).astype(int)
    qpoints_full = np.round(qpoints_full*mesh)/mesh
    def qpoints_correction(q): # Only used if the FBZ is a layered hexagon
        qx,qy,qz = q.T
        s = np.maximum(qx,qy)>(1-qx-qy)
        b = np.minimum(qx,qy)>(2-qx-qy)
        lr = (qx>=qy)
        qx[s&lr|b] -= 1
        qy[s&~lr|b] -= 1
        qz[qz>0.5] -= 1
        q_correct = np.c_[qx,qy,qz]
        return q_correct
    qpoints_full = qpoints_correction(qpoints_full)
    q = qpoints_full @ RecLatt # nm-1
    
    # Full mesh
    omega_full = omega[full]
    v = v_full.T.reshape((3,Nq,Ns),order='F')
    F = F_final.T.reshape((3,Nq,Ns),order='F')/(omega_full+eps) # nm
    w_ph = w_ph.reshape((Nqired,Ns),order='F')[full]
    w_ph_N = w_ph_N.reshape((Nqired,Ns),order='F')[full]
    
    # Relaxation time
    #Directions = np.atleast_1d(eta_Dims)
    v_Dims = np.linalg.norm(v[eta_Dims],axis=0)
    w_boundary = 2*v_Dims/(D_boundary*1e9)
    tau_ph = 1/(w_ph+eps)
    tau = 1/(w_ph+w_boundary+eps)
    Nratio = w_ph_N*tau_ph
    
    # Capacity & Conductivity
    Latt = 2*np.pi*np.linalg.inv(RecLatt).T # nm
    Vol = np.abs(Latt[0] @ np.cross(Latt[1],Latt[2]))*1e3 # A^3
    from scipy.constants import hbar; hbar *= 1e12 # J*ps
    from scipy.constants import k as kb # J/K
    x = hbar*omega_full/kb/T0/2
    c = kb*x**2/np.sinh(x+eps)**2/Vol/Nq*1e24 # J/cm^3
    kappa_cvF = np.einsum('mn,imn,jmn->ij',c,v,F)
    kappa_RTA_bulk = np.einsum('mn,imn,jmn->ij',c*tau_ph,v,v)
    kappa_RTA = np.einsum('mn,imn,jmn->ij',c*tau,v,v)
    
    # Correct Nratio
    # Assert tensors are diagonal
   
    q_omega = q.T[:,:,None]/(omega_full+eps)
    kappa_u = np.diag(kappa_cvF-kappa_RTA_bulk) #drifting component
    def corrected_paramenters(Nratio,zeta=0.9):
        Pij = np.einsum('mn,imn,jmn->ij',Nratio*c,q_omega[eta_Dims],v[eta_Dims])
        Gij = np.einsum('mn,imn,jmn->ij',(1-Nratio)*Nratio*c*w_ph,q_omega[eta_Dims],q_omega[eta_Dims])
        pij = np.einsum('mn,imn,jmn->ij',2*c,q_omega[eta_Dims],v[eta_Dims])
        gij = np.einsum('mn,imn,jmn->ij',(1-2*Nratio)*c*w_ph,q_omega[eta_Dims],q_omega[eta_Dims])
        # kappa_uij = Pij.T @ np.linalg.inv(Gij) @ Pij
        # kappa_uij = np.einsum('ji,jk,kl->il',Pij,np.linalg.inv(Gij),Pij)
        kappa_uij = np.diag(Pij**2)/np.diag(Gij)
        delta_Nr = np.diag(Pij**2-Gij*kappa_u[eta_Dims])/np.diag(gij*kappa_u[eta_Dims]-Pij*pij)
        Nr = Nratio+np.mean(delta_Nr)*zeta
        np.clip(Nr,0,1,out=Nr)
        return Nr,kappa_uij
    Nr,kappa_uij = corrected_paramenters(Nratio)
    for i in range(30):
        Nr,kappa_uij = corrected_paramenters(Nr)
        if np.linalg.norm(kappa_uij/kappa_u[eta_Dims]-1)<0.01:
            break
    
    kappa_u = np.diag(kappa_u)
    for i in eta_Dims:
        kappa_u[i,i] = kappa_uij[i]
    
    # Output
    from scipy.constants import elementary_charge as E_charge
    w_qgrid = phonon.get_mesh_dict()['weights']
    # qpoints_full = qpoints_full_list[:,2:]
    # qpoints_full[qpoints_full>0.5] -= 1
    # qpoints_full[qpoints_full<-0.5] += 1
    if calc_fullmesh:
        omega = omega_full
    else:
        Nq = Nqired
        c = c[ired]*w_qgrid[:,None]
        v = v[:,ired]
        tau = tau[ired]
        F = v*tau
        Nr = Nr[ired] 
    Vec_freqs = omega.flatten()/2/np.pi # THz
    Vec_cqs = c.flatten()/E_charge*1e-24 # ph_w*eV/A^3/K
    Vec_vqs = v.reshape(3,Nq*Ns)*10 # A/ps
    Vec_Fsc_qs = F.reshape(3,Nq*Ns)*1e-9 # m
    Vec_tausc_qs = tau.flatten() # ps
    Vec_tauNsc_qs = (tau/(Nr+eps)).flatten()
    Vec_tauRsc_qs = (tau/(1-Nr+eps)).flatten()
    Nratio_qs = Nr.flatten()
    
    kappa_RTA_qs = np.zeros((3,3,len(Vec_cqs)))
    
    for i in np.arange(3):
        for j in np.arange(3):
            kappa_RTA_qs[i,j] = Vec_cqs*Vec_vqs[i]*Vec_vqs[j]*Vec_tausc_qs
    
    if Gamma_cut == False:
        return qpoints_full,Vec_freqs,Vec_cqs,Vec_vqs,Vec_Fsc_qs,Vec_tausc_qs,Vec_tauNsc_qs,Vec_tauRsc_qs,kappa_RTA,kappa_u,kappa_RTA_qs,Nratio_qs 
    else:
        return qpoints_full,Vec_freqs[3:],Vec_cqs[3:],Vec_vqs[:,3:],Vec_Fsc_qs[:,3:],Vec_tausc_qs[3:],Vec_tauNsc_qs[3:],Vec_tauRsc_qs[3:],kappa_RTA,kappa_u,kappa_RTA_qs[3:],Nratio_qs[3:]

    
def load_ShengBTE_Phonons(T0,phonon,Dir_BTE_HarPhons,Dir_BTE_MFD,Dir_BTE_lifetime,D_boundary,is_isotope,four_phonon=False,calc_fullmesh=False,tau_Dims=[0,1,2],Gamma_cut=False):
    eps = 1e-50
    
    w_qgrid = phonon.get_mesh_dict()['weights']
    (Nqired,Ns) = phonon.get_mesh_dict()['frequencies'].shape #irreducible.
    Wmat_grid = np.reshape(np.repeat(w_qgrid,Ns).T,(Nqired,Ns))

    freqs_ired = np.loadtxt(Dir_BTE_HarPhons+'BTE.omega')/2/np.pi # in THz
    qpoints_full_list = np.loadtxt(Dir_BTE_HarPhons+'BTE.qpoints_full')
    qpoints_list = np.loadtxt(Dir_BTE_HarPhons+'BTE.qpoints')

    ired = qpoints_list[:,1].astype(int)-1
    full = qpoints_full_list[:,1].astype(int)-1
    Nq,(Nqired,Ns) = len(full),freqs_ired.shape
    qpoints_full = qpoints_full_list[:,2:]
    mesh = np.round(1/(1-qpoints_full[-1])).astype(int)
    qpoints_full = np.round(qpoints_full*mesh)/mesh
    def qpoints_correction(q): # Only used if the FBZ is a layered hexagon
        qx,qy,qz = q.T
        s = np.maximum(qx,qy)>(1-qx-qy)
        b = np.minimum(qx,qy)>(2-qx-qy)
        lr = (qx>=qy)
        qx[s&lr|b] -= 1
        qy[s&~lr|b] -= 1
        qz[qz>0.5] -= 1
        q_correct = np.c_[qx,qy,qz]
        return q_correct
    qpoints_full = qpoints_correction(qpoints_full)
    
    
    
    # qpoints_full = qpoints_full_list[:,2:]
    
    # # wrap into the 1st BZ.
    # qpoints_full[qpoints_full>0.5] -=1
    # qpoints_full[qpoints_full<-0.5] +=1
    

    if calc_fullmesh:
        freqs = expand_qired2qfull(freqs_ired,qpoints_full_list)
    else:
        freqs = freqs_ired

    (Nq,Ns) = freqs.shape # if calc_fullmesh = True, then Nq = total q points.
    #Nmodes = Nq*Ns

    # calculate at qpoints 
    cqseV = mode_cv(T0,freqs*THzToEv) 
    cqseV[np.isinf(cqseV)] = 0
    cqseV[np.isnan(cqseV)]  = 0

    mesh = phonon._mesh.get_mesh_numbers()

    if calc_fullmesh:
        cqs = cqseV/np.prod(mesh)/phonon._unitcell.get_volume()   
        gv = import_BTE_PhvecProps(Dir_BTE_HarPhons+'BTE.v_full',Ns)*10 # Angstrom THz
        gvi = import_BTE_PhvecProps(Dir_BTE_HarPhons+'BTE.v',Ns)*10 # Angstrom THz

    else:
        gv = import_BTE_PhvecProps(Dir_BTE_HarPhons+'BTE.v',Ns)*10 # group velocity at irred wedge. 
        cqs = cqseV*Wmat_grid/np.prod(mesh)/phonon._unitcell.get_volume() # multiply the qpoint weights

    # flatten gv of shape (3,Nq,Ns) to (3,Nq*Ns)
    Vec_vqs = np.zeros((3,Nq*Ns))
    
     
    
    for i in range(3):
        Vec_vqs[i] = Vectorize_mode_props(gv[i]) 

    scatt_rate_ph = np.loadtxt(Dir_BTE_lifetime+'BTE.w_3ph')[:,-1] # in ps^-1
    
    if calc_fullmesh:
        speed_qs = np.sqrt(gvi[0]**2+gvi[1]**2+gvi[2]**2) #Angstrom THz
    else:
        speed_qs = np.sqrt(gv[0]**2+gv[1]**2+gv[2]**2) #Angstrom THz
    
    scatt_rate_b = 2*speed_qs/(D_boundary/Angstrom) #scattering rate ps^-1
    #print(scatt_rate_b)
    
    #print(four_phonon)
    
    if four_phonon:
        #print('reading 4ph scatt rate')
        scatt_rate_ph += np.loadtxt(Dir_BTE_lifetime+'BTE.w_4ph')[:,-1]
        
    scatt_rate_N = np.loadtxt(Dir_BTE_lifetime+'BTE.w_3ph_Normal')[:,-1] 
    scatt_rate_U = np.loadtxt(Dir_BTE_lifetime+'BTE.w_3ph_Umklapp')[:,-1] 
    
    if four_phonon:
        #print('reading 4ph N & U scatt rate')
        scatt_rate_N += np.loadtxt(Dir_BTE_lifetime+'BTE.w_4ph_Normal')[:,-1] 
        scatt_rate_U += np.loadtxt(Dir_BTE_lifetime+'BTE.w_4ph_Umklapp')[:,-1]

    if is_isotope:
        scatt_rate_I = np.loadtxt(Dir_BTE_HarPhons+'BTE.w_isotopic')[:,1]
        scatt_rate_I = np.reshape(scatt_rate_I,freqs_ired.T.shape).T
    else:
        scatt_rate_I = 0.0


    scatt_rate_ph = np.reshape(scatt_rate_ph,freqs_ired.T.shape).T
    scatt_rate_N = np.reshape(scatt_rate_N,freqs_ired.T.shape).T
    scatt_rate_U =  np.reshape(scatt_rate_U,freqs_ired.T.shape).T

    
    
    tau_ph = 1./(scatt_rate_N+scatt_rate_U)
    tau_N = 1./scatt_rate_N # ps.
    tau_R = 1/(scatt_rate_U+scatt_rate_I+eps+scatt_rate_b) # ps
    
    tau_qs = 1./(scatt_rate_N+scatt_rate_U+scatt_rate_I+scatt_rate_b)

    tau_N[tau_N>THz] = 0; tau_N[np.isnan(tau_N)] = 0
    tau_R[tau_R>THz] = 0; tau_R[np.isnan(tau_R)] = 0
    tau_qs[tau_qs>THz] = 0; tau_qs[np.isnan(tau_qs)] = 0


    if calc_fullmesh:
        # expand lifetimes to qpoints_full
        
        tau_qs_ired = tau_qs
        
        tau_N = expand_qired2qfull(tau_N,qpoints_full_list)
        tau_R = expand_qired2qfull(tau_R,qpoints_full_list)
        tau_ph = expand_qired2qfull(tau_ph, qpoints_full_list)
        tau_qs = expand_qired2qfull(tau_qs,qpoints_full_list)
        
        
        
        #Vec_Rscatt_rate = expand_qired2qfull(scatt_rate_I,qpoints_full_list)


    Vec_freqs = Vectorize_mode_props(freqs)
    Vec_cqs = Vectorize_mode_props(cqs)
    Vec_tauN_qs = Vectorize_mode_props(tau_N)
    Vec_tauR_qs = Vectorize_mode_props(tau_R)
    Vec_tau_qs = Vectorize_mode_props(tau_qs)
    #Vec_tauI_qs = Vectorize_mode_props(tau_I)

    F_modes_full = import_BTE_PhvecProps(Dir_BTE_MFD+'BTE.F_final',Ns) # nm*Trad/s
    

    
    
    if calc_fullmesh:
        F_modes = F_modes_full/(eps+freqs*2*np.pi)*10 # full mesh. Angstrom.
        Vec_Fqs = np.zeros((3,Nq*Ns))    
        for i in range(3):
            Vec_Fqs[i] = Vectorize_mode_props(F_modes[i])
        
        Vec_MFP = np.sqrt(np.sum(Vec_Fqs*Vec_Fqs,axis=0))
        SupFunc = 1/(1+Vec_MFP/(D_boundary/Angstrom))
           
        Vec_SupFunc_reap = np.broadcast_to(SupFunc,(3,)+SupFunc.shape)
        
        Vec_Fsc_qs = Vec_Fqs*Vec_SupFunc_reap
        
        
        # this is performed on irreducible BZ.
        Vec_vFqs_ired,Vec_Fsc_qs_ired,Vec_tausc_qs_ired = vFmodes_full_to_irep(F_modes_full,D_boundary,gv,gvi,tau_qs_ired,freqs_ired,qpoints_full_list,tau_Dims)
        
        tausc_qs_ired = np.reshape(Vec_tausc_qs_ired, freqs_ired.shape)
        tausc_qs = expand_qired2qfull(tausc_qs_ired,qpoints_full_list)
        Vec_tausc_qs = Vectorize_mode_props(tausc_qs)
        
        
        # for i in range(3):
        #     Vec_Fsc_qs[i] = np.sqrt(np.abs(Vec_Fqs[i]*Vec_vqs[i]*Vec_tausc_qs))*np.sign(Vec_Fqs[i]) #Angstrom

    else: 
        gvf = import_BTE_PhvecProps(Dir_BTE_HarPhons+'BTE.v_full',Ns)*10 # convert to Angstrom*THz
        Vec_vFqs,Vec_Fsc_qs,Vec_tausc_qs = vFmodes_full_to_irep(F_modes_full,D_boundary,gvf,gv,tau_qs,freqs_ired,qpoints_full_list,tau_Dims)
        
    
    
    Nratio_qs = (Vec_tau_qs+eps)/(Vec_tauN_qs+eps)
    Rratio_qs = (Vec_tau_qs+eps)/(Vec_tauR_qs+eps)
    
    Nratio_qs[Nratio_qs>THz] = 0.0
    Rratio_qs[Rratio_qs>THz] = 0.0
    
    Vec_tauNsc_qs = (Vec_tausc_qs)/(Nratio_qs+eps)
    Vec_tauRsc_qs = (Vec_tausc_qs)/(Rratio_qs+eps)
    
    Vec_tauNsc_qs[Vec_tauNsc_qs>THz] = Vec_tauN_qs[Vec_tauNsc_qs>THz]
    Vec_tauRsc_qs[Vec_tauRsc_qs>THz] = Vec_tauR_qs[Vec_tauRsc_qs>THz]
    Vec_tausc_qs[Vec_tausc_qs>THz] = Vec_tau_qs[Vec_tausc_qs>THz]
    #Vec_Fsc_qs = Vec_vqs*Vec_tau_qs # 
    Vec_Fsc_qs*= Angstrom # MFD convert to m.
    
    kappa_cvF = np.zeros((3,3))
    
    kappa_qs_eV = np.zeros((3,3,len(Vec_cqs)))
    
    if calc_fullmesh:
        for (i,j) in [[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]]:
            kappa_cvF[i,j] = np.sum(Vec_vqs[i]*Vec_Fqs[j]*Vec_cqs)* (EV/Angstrom**3)*(Angstrom**2*THz)
            kappa_cvF[j,i] = kappa_cvF[i,j]    
            kappa_qs_eV[i,j] = Vec_vqs[i]*Vec_Fqs[j]*Vec_cqs
            kappa_qs_eV[j,i] = kappa_qs_eV[i,j]
    else:
        Gamma = 1/Vec_tausc_qs
        Gamma[np.isnan(Gamma)]=0
        Gamma[np.isinf(Gamma)]=0
        # print(Gamma)
        for (i,j) in [[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]]:
            
            # kappa_cvF[i,j] = np.sum(Vec_Fsc_qs[i]*Vec_Fsc_qs[j]*Gamma*Vec_cqs)* (EV/Angstrom**3)*(THz) # RTA np.sum(Vec_cqs*Vec_vqs[i]*Vec_vqs[j]*Vec_tau_qs)* (EV/Angstrom**3)*(Angstrom*THz*Angstrom) #
            kappa_cvF[i,j] = np.sum(Vec_vFqs[i,j]*Vec_cqs)* (EV/Angstrom**3)*(Angstrom**2*THz)
            kappa_cvF[j,i] = kappa_cvF[i,j]
            kappa_qs_eV[i,j] = Vec_vFqs[i,j]*Vec_cqs
            kappa_qs_eV[j,i] = kappa_qs_eV[i,j]
            
            
    kappa_cvF = symmetrize_kappa(kappa_cvF, phonon)
    
    if Gamma_cut == False:
        return qpoints_full,Vec_freqs,Vec_cqs,Vec_vqs,Vec_Fsc_qs,Vec_tausc_qs,Vec_tauNsc_qs,Vec_tauRsc_qs,kappa_cvF,kappa_qs_eV,Nratio_qs 
    else:
        return qpoints_full,Vec_freqs[3:],Vec_cqs[3:],Vec_vqs[:,3:],Vec_Fsc_qs[:,3:],Vec_tausc_qs[3:],Vec_tauNsc_qs[3:],Vec_tauRsc_qs[3:],kappa_cvF,kappa_qs_eV[:,:,3:],Nratio_qs[3:] # when using the full scatt matrix, remove the 0,1,2 th modes.

# --------------------------------------- Compute thermal conductivity from modal properties --------------------------------------------# 

# Calculate Callay thermal conductivity
def symmetrize_kappa(kappa,phonon):
    rots = phonon.symmetry.get_symmetry_operations()['rotations']
    Nrots = len(rots)
    
    kappa_sym = np.zeros((3,3))
    
    for rot in rots:
        invrot = np.linalg.inv(rot)
        
        kappa_sym += np.dot(np.dot(rot,kappa),invrot)/Nrots
        
    return kappa_sym
    
    
def get_kappa_callaway(Vec_cqs,Vec_freqs,Vec_tauN_qs,Vec_tauR_qs,Vec_vqs,phonon,rots_qpoints):
    eps = 1e-50
    units_factor = EV*THz/Angstrom 

   # kappa = np.zeros(6) # xx yy zz yz xz xy

    Vec_eqs = Vec_freqs*Hbar #EV

    Vec_tau_qs = 1/(1/(eps+Vec_tauN_qs)+1/(eps+Vec_tauR_qs))

    qpoints = phonon.get_mesh_dict()['qpoints']

    reclat = np.linalg.inv( phonon.get_unitcell().get_cell())*2*np.pi # reciprocal lattice in Angstrom^-1

    Nq = len(qpoints)
    Nmodes =len(Vec_cqs)
    Ns = int(Nmodes/Nq)
    
    Vec_vv_qs = np.zeros((6,)+Vec_vqs.shape[1:],dtype=Vec_vqs.dtype)
    
    Ivec = np.zeros(3)
    #IItensor = np.zeros(6)
    Jtensor = np.zeros(6)
    
    for iq in range(Nq):
        
        qfrac = qpoints[iq]
        
        q = np.dot(reclat,qfrac) # Angstrom^-1        
        rots_sitesym = rots_qpoints[iq]
        
        multi = len(rots_sitesym)
        
        vs_at_q = Vec_vqs[:,iq*Ns:(iq+1)*Ns] # 3-by_Ns 
        cs_at_q = Vec_cqs[iq*Ns:(iq+1)*Ns]
        es_at_q = Vec_eqs[iq*Ns:(iq+1)*Ns]
        tau_R_at_q = Vec_tauR_qs[iq*Ns:(iq+1)*Ns]
        tau_N_at_q = Vec_tauN_qs[iq*Ns:(iq+1)*Ns]
        tau_at_q = Vec_tau_qs[iq*Ns:(iq+1)*Ns]
        
        Cqv_x = vs_at_q[0]*q[0]*cs_at_q
        Cqv_y = vs_at_q[1]*q[1]*cs_at_q
        Cqv_z = vs_at_q[2]*q[2]*cs_at_q
        
        Cqv_at_q = np.array([Cqv_x,Cqv_y,Cqv_z])
    
        # (Angstrom^-2*EV/K/Angstrom^3 )/EV/(THz*EV) = (EV*THz/Ansgrom^5/K)
        
        Ivec[0]+= np.sum(Cqv_at_q[0]/(es_at_q+eps)*(tau_at_q+eps)/(tau_N_at_q+eps))
        Ivec[1]+= np.sum(Cqv_at_q[1]/(es_at_q+eps)*(tau_at_q+eps)/(tau_N_at_q+eps))
        Ivec[2]+= np.sum(Cqv_at_q[2]/(es_at_q+eps)*(tau_at_q+eps)/(tau_N_at_q+eps))
        
        gv_by_gv_at_q = np.zeros((6,)+vs_at_q.shape[1:],dtype=vs_at_q.dtype)
        
        q_by_q = np.zeros(6,dtype=q.dtype)
            
        for idir,(i,j) in enumerate([[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]]):
            
            for rot in rots_sitesym:
                r_cart = similarity_transformation(reclat, rot)                                
                r_vs_at_q = np.dot(r_cart,vs_at_q)                
                
                #r_q = np.dot(r_cart,q)                
                gv_by_gv_at_q[idir] += r_vs_at_q[i]*r_vs_at_q[j]                
                
                #q_by_q[idir] +=  r_q[i]*r_q[j]

            gv_by_gv_at_q[idir] /= multi # symmetrize gvm_by_gvm
            #q_by_q[idir] /=multi
            
            q_by_q[idir] = q[i]*q[j]
            
            Jtensor[idir]+= q_by_q[idir]*np.sum((cs_at_q+eps)/(es_at_q**2+eps)*(tau_at_q+eps)
                                                /(tau_N_at_q*tau_R_at_q+eps))
            
     
        Vec_vv_qs[:,iq*Ns:(iq+1)*Ns] = gv_by_gv_at_q
            
    
    kappa_N = np.zeros((3,3))           
    kappa_RTA = np.zeros((3,3))

    
    
    for ipair,(a,b) in enumerate([[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]]):
        kappa_N[a,b] = Ivec[a]*Ivec[b]/(Jtensor[ipair]+eps)*units_factor 
        kappa_N[b,a] = kappa_N[a,b]
        
        kappa_RTA[a,b] = np.sum(Vec_cqs*Vec_vv_qs[ipair]*Vec_tau_qs)*units_factor 
        # EV/K/Angstrom^3 * THz^2*Angstrom^2/THz = EV*THz/Angstrom/K        
        kappa_RTA[b,a] = kappa_RTA[a,b]    


    kappa_N = symmetrize_kappa(kappa_N,phonon)
    kappa_RTA = symmetrize_kappa(kappa_RTA,phonon)
    kappa_callaway = kappa_RTA+kappa_N


    return kappa_RTA,kappa_callaway

def get_kappa_callaway_fullq(Vec_cqs,Vec_freqs,Vec_tauN_qs,Vec_tauR_qs,Vec_vqs,qpoints_full,phonon):
    eps = 1e-50
    units_factor = EV*THz/Angstrom 

   # kappa = np.zeros(6) # xx yy zz yz xz xy

    Vec_eqs = Vec_freqs*Hbar #EV

    Vec_tau_qs = 1/(1/(eps+Vec_tauN_qs)+1/(eps+Vec_tauR_qs))

    #qpoints = phonon.get_mesh_dict()['qpoints']

    reclat = np.linalg.inv( phonon.get_unitcell().get_cell())*2*np.pi # reciprocal lattice in Angstrom^-1

    Nq = len(qpoints_full)
    Nmodes =len(Vec_cqs)
    Ns = int(Nmodes/Nq)
    
    Vec_vv_qs = np.zeros((6,)+Vec_vqs.shape[1:],dtype=Vec_vqs.dtype)
    
    Ivec = np.zeros(3)
    #IItensor = np.zeros(6)
    Jtensor = np.zeros(6)
    
    for iq in range(Nq):
        
        qfrac = qpoints_full[iq]
        
        q = np.dot(reclat,qfrac) # Angstrom^-1        
        
        vs_at_q = Vec_vqs[:,iq*Ns:(iq+1)*Ns] # 3-by_Ns 
        cs_at_q = Vec_cqs[iq*Ns:(iq+1)*Ns]
        es_at_q = Vec_eqs[iq*Ns:(iq+1)*Ns]
        tau_R_at_q = Vec_tauR_qs[iq*Ns:(iq+1)*Ns]
        tau_N_at_q = Vec_tauN_qs[iq*Ns:(iq+1)*Ns]
        tau_at_q = Vec_tau_qs[iq*Ns:(iq+1)*Ns]
        
        Cqv_x = vs_at_q[0]*q[0]*cs_at_q
        Cqv_y = vs_at_q[1]*q[1]*cs_at_q
        Cqv_z = vs_at_q[2]*q[2]*cs_at_q
        
        Cqv_at_q = np.array([Cqv_x,Cqv_y,Cqv_z])
    
        # (Angstrom^-2*EV/K/Angstrom^3 )/EV/(THz*EV) = (EV*THz/Ansgrom^5/K)
        
        Ivec[0]+= np.sum(Cqv_at_q[0]/(es_at_q+eps)*(tau_at_q+eps)/(tau_N_at_q+eps))
        Ivec[1]+= np.sum(Cqv_at_q[1]/(es_at_q+eps)*(tau_at_q+eps)/(tau_N_at_q+eps))
        Ivec[2]+= np.sum(Cqv_at_q[2]/(es_at_q+eps)*(tau_at_q+eps)/(tau_N_at_q+eps))
        
        gv_by_gv_at_q = np.zeros((6,)+vs_at_q.shape[1:],dtype=vs_at_q.dtype)
        
        q_by_q = np.zeros(6,dtype=q.dtype)
            
        for idir,(i,j) in enumerate([[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]]):
            
            gv_by_gv_at_q[idir] = vs_at_q[i]*vs_at_q[j]
            
            q_by_q[idir] = q[i]*q[j]
            
            Jtensor[idir]+= q_by_q[idir]*np.sum((cs_at_q+eps)/(es_at_q**2+eps)*(tau_at_q+eps)
                                                /(tau_N_at_q*tau_R_at_q+eps))
            
     
        Vec_vv_qs[:,iq*Ns:(iq+1)*Ns] = gv_by_gv_at_q
            
    
    kappa_N = np.zeros((3,3))           
    kappa_RTA = np.zeros((3,3))

    
    for ipair,(a,b) in enumerate([[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]]):
        kappa_N[a,b] = Ivec[a]*Ivec[b]/(Jtensor[ipair]+eps)*units_factor 
        kappa_N[b,a] = kappa_N[a,b]
        
        kappa_RTA[a,b] = np.sum(Vec_cqs*Vec_vv_qs[ipair]*Vec_tau_qs)*units_factor 
        # EV/K/Angstrom^3 * THz^2*Angstrom^2/THz = EV*THz/Angstrom/K        
        kappa_RTA[b,a] = kappa_RTA[a,b]    


    kappa_N = symmetrize_kappa(kappa_N,phonon)
    kappa_RTA = symmetrize_kappa(kappa_RTA,phonon)
    kappa_callaway = kappa_RTA+kappa_N


    return kappa_RTA,kappa_callaway
    
    
# Get rotational symmetry of irreducible qpoint    
def get_qpoint_rotsym(phonon):
    reclat = np.linalg.inv( phonon.get_unitcell().get_cell())
    rots =  phonon.symmetry.reciprocal_operations
    qpoints = phonon.get_mesh_dict()['qpoints']
    
    rots_qpoints = []
    #multi_qpoints= []

    
    for iq,q in enumerate(qpoints):
        q = qpoints[iq]
    
        #multi = 0
        rots_sitesysm = []
        
        for rot in rots:
            q_rot = np.dot(rot,q)
            diff = q - q_rot
            diff -= np.rint(diff)
            dist = np.linalg.norm(np.dot(reclat, diff))
            if np.abs(dist) < phonon.symmetry.tolerance:
                #multi += 1
                rots_sitesysm.append(rot)
                
                
        #multi_qpoints.append(multi)
        rots_qpoints.append(rots_sitesysm)
       
    return rots_qpoints
                


# compute mean relaxation times for N and R scatterings, and the collective transpor factor.
def get_aveNR_Relxtime(Vec_cqs,Vec_tauN_qs,Vec_tauR_qs):
    ave_tauN = np.sum(Vec_cqs*Vec_tauN_qs)/np.sum(Vec_cqs)
    ave_tauR = np.sum(Vec_cqs*Vec_tauR_qs)/np.sum(Vec_cqs)
    Nratio = 1/(1+ave_tauN/ave_tauR)
    
    return ave_tauN,ave_tauR,Nratio

# --------------------------------------- Susceptibilities relating population deviation and phonon generation --------------------------------------------#        

def X1D_modes_at_q(XIx,OMEGA,Fs_at_q,taus_at_q,x_direct=0,Xdtype='complex64'):
    '''
    this function calculates the modal susceptibility on the 2D meshgrid of fourier space.
    In-plane directions are Symmtrized. 
    '''
    Nx = XIx.shape[0]
    Nw = OMEGA.shape[1]
    Ns = len(taus_at_q)    
    Xmodes_1D_q = np.zeros((Ns,Nx,Nw),dtype=Xdtype)
    Xqmodes_1D_q = np.zeros((Ns,Nx,Nw),dtype=Xdtype)
    
    ix = x_direct

    
    for s in range(Ns):
        Fx = Fs_at_q[ix,s]
        
        FXi = Fx*XIx        
        jWt = 1j*OMEGA*taus_at_q[s]
        
        a= 1+jWt
         
        Xs = a/(a**2+FXi**2)
        Xq_s = -1j*FXi*Xs/a
        

        Xs[np.isnan(Xs)] = 1.0
        Xq_s[np.isnan(Xq_s)] = 0.0
        
        Xmodes_1D_q[s] = Xs
        Xqmodes_1D_q[s] = Xq_s
        
        
    return Xmodes_1D_q,Xqmodes_1D_q


def X1Diso_modes_at_q(XIx,OMEGA,Fs_at_q,taus_at_q,r_directs=(0,1),Xdtype='complex64'):
    '''
    this function symmetrize the susceptibility using in-plane 2D isometry.
    '''
    Nr = XIx.shape[0]
    
    Nw = OMEGA.shape[1]
    
    Ns = len(taus_at_q)
    
    ir1 = r_directs[0]
    ir2 = r_directs[1]
    #iz = rz_directs[2]
    
    Xmodes_iso_q = np.zeros((Ns,Nr,Nw),dtype=Xdtype)
    Xqmodes_iso_q = np.zeros((Ns,Nr,Nw),dtype=Xdtype)    
    
    for s in range(Ns):
        F = Fs_at_q[:,s]
        
        Fx1 =  F[ir1] 
        Fx2 =  F[ir2]
        Fr = np.sqrt(Fx1**2+Fx2**2)
        
        Wt = OMEGA*taus_at_q[s]
        a = (1+1j*Wt)
        b = Fr*XIx
    
        Xs = 1/np.sqrt(a**2+b**2)
 
        Xq_s =-2j/np.pi*np.arctanh(b/np.sqrt(a**2+b**2))/np.sqrt(a**2+b**2)
        
        Xs[np.isnan(Xs)] = 1.0
        Xq_s[np.isnan(Xq_s)] = 0.0

        Xmodes_iso_q[s] = Xs
        Xqmodes_iso_q[s] =  Xq_s
        
    return Xmodes_iso_q,Xqmodes_iso_q


def X2Dxy_modes_at_q(XIx,XIy,OMEGA,Fs_at_q,taus_at_q,xy_directs=(0,1),Xdtype='complex64'):
    '''
    this function symmetrize the susceptibility using in-plane 2D isometry.
    '''
    Nx = XIx.shape[0]
    Ny = XIy.shape[1]
    Nw = OMEGA.shape[2]
    
    Ns = len(taus_at_q)
    
    ir1 = xy_directs[0]
    ir2 = xy_directs[1]

    
    Xmodes_q = np.zeros((Ns,Nx,Ny,Nw),dtype=Xdtype)
    Xqmodes_q = np.zeros((Ns,Nx,Ny,Nw),dtype=Xdtype)    
    
    for s in range(Ns):
        F = Fs_at_q[:,s]
        
        Fx = F[ir1]
        Fy = F[ir2]
        
        jWt = 1j*OMEGA*taus_at_q[s]
       
        FXi = Fx*XIx + Fy*XIy
        
        FXim = Fx*XIx - Fy*XIy # mirror with x.

        
        a = (1+jWt)
        
        Xs = 0.5*( a/(a**2+FXi**2) +a/(a**2+FXim**2))

        Xq_s = -1j*FXi*Xs/a

        Xs[np.isnan(Xs)] = 1.0
        Xq_s[np.isnan(Xq_s)] = 0.0

        Xmodes_q[s] = Xs
        Xqmodes_q[s] =  Xq_s
        
    return Xmodes_q,Xqmodes_q



def X2Dxz_modes_at_q(XIx,XIz,OMEGA,Fs_at_q,taus_at_q,xz_directs=(0,2),Xdtype='complex64'):

    '''
    this function calculates the modal susceptibility on the 2D meshgrid of fourier space.
    default is x and z direction.
    '''
    Nx = XIx.shape[0]
    Nz = XIz.shape[1]
    Nw = OMEGA.shape[2]
    
    Ns = len(taus_at_q)
    ix = xz_directs[0]
    iz = xz_directs[1]
    
    Xmodes_2D_q = np.zeros((Ns,Nx,Nz,Nw),dtype=Xdtype)

    
    for s in range(Ns):
        F = Fs_at_q[:,s]
        FXi = F[ix]*XIx+F[iz]*XIz  

        Wt = OMEGA*taus_at_q[s]
        
        Xs = 1/(1+1j*Wt+1j*FXi)
    
        Xmodes_2D_q[s] = Xs
        
    return Xmodes_2D_q

def Xcyln_modes_at_q(XIr,XIz,OMEGA,Fr_at_q,Fz_at_q,taus_at_q,Xdtype='complex64'):
    '''
    this function calculates the modal susceptibility on the cylindrical meshgrid of fourier space.
    '''
    Nr = XIr.shape[0]
    Nz = XIz.shape[1]
    Nw = OMEGA.shape[2]
    
    Ns = len(taus_at_q)
    

    Xmodes_cyln_q = np.zeros((Ns,Nr,Nz,Nw),dtype=Xdtype)
    Xqmodes_cyln_q = np.zeros((Ns,Nr,Nz,Nw),dtype=Xdtype)

    
    for s in range(Ns):
        Fr = Fr_at_q[s]
        Fz = Fz_at_q[s]

        Wt = OMEGA*taus_at_q[s]
        
        a = 1+1j*Wt
        b = Fr*XIr
        c = Fz*XIz                
        
        Den = np.sqrt((a+1j*c)**2+b**2)
        
        Xs = 1/Den
        Xs[np.isnan(Xs)] = 1.0
        
        Xq_s = -2.0j/np.pi/Den*np.arctan(b/Den)
        Xq_s[np.isnan(Xq_s)] = 0.0
        
        Xmodes_cyln_q[s] = Xs
        Xqmodes_cyln_q[s] = Xq_s
        
    return Xmodes_cyln_q,Xqmodes_cyln_q


def X3D_modes_at_q(XIx,XIy,XIz,OMEGA,Fs_at_q,taus_at_q,Xdtype='complex64'):
    '''
    this function calculates the modal susceptibility on the 3D meshgrid of fourier space.
    '''
    #eps = 1e-50
    Nx = XIx.shape[0]
    Ny = XIy.shape[1]
    Nz = XIz.shape[2]
    Nw = OMEGA.shape[3]
    
    Ns = len(taus_at_q)
    
    Xmodes_3D_q = np.zeros((Ns,Nx,Ny,Nz,Nw),dtype=Xdtype)
    
    for s in range(Ns):
        F = Fs_at_q[:,s]
        FXi = F[0]*XIx+F[1]*XIy+F[2]*XIz
        Wt = OMEGA*taus_at_q[s]
        Xs = 1/(1+1j*FXi+1j*Wt)
        Xmodes_3D_q[s] = Xs
        
    
    return Xmodes_3D_q


 # --------------------------------------- Green's function of temperature rise in Fourier heat conduction----------------------------------------------------#

# Isotropic or 1D Fourier Green's function 
def GF_Fourier_1D(XI,OMEGAH_Trads,kappa_s,C,Q0=1):
    return Q0/(kappa_s*XI*XI+1j*OMEGAH_Trads*C*THz)
    
# Fourier Green's function with cylindrical anisotropy    
def GF_Fourier_2D(XIr,XIz,OMEGAH_Trads,kappa_r,kappa_z,C,Q0=1):
    return Q0/(kappa_r*XIr*XIr+kappa_z*XIz*XIz+1j*OMEGAH_Trads*C*THz)
    
def GF_Fourier_3D(XIx,XIy,XIz,OMEGAH_Trads,kappa_x,kappa_y,kappa_z,C,Q0=1):
    return Q0/(kappa_x*XIx*XIx+kappa_y*XIy*XIy+kappa_z*XIz*XIz+1j*OMEGAH_Trads*C*THz)

 # --------------------------------------- Solve linearized callay model for T rise and drift velocity --------------------------------------------------------#


def Phonon_GenRate(Vec_cqs,Vec_freqs,Q0=1):
    #eps = 1e-50
    #e_qs = Vec_freqs*THzToEv #*EV # energy of phonon modes 
    Q_qs = Vec_cqs*Q0/np.sum(Vec_cqs)/(EV/Angstrom**3*THz)
    
    return Q_qs

def Solve1D_TempRN_udrift(XIx,OMEGAH,Vec_cqs,Vec_freqs,Vec_Fsc_qs,Vec_tauN_qs,Vec_tauR_qs,Vec_tau_qs,T0,phonon,rots_qpoints,x_direct=0,Xdtype='complex64'):
    # xz_directs specifies in-plane and cross-plane directions.
    eps = 1e-50
    
    ix = x_direct #r_directs[0]
    #iy = r_directs[1]
    
    
    Nx = XIx.shape[0]
    Nw = OMEGAH.shape[1]
    
    mesh_shape = (Nx,Nw)
    
    
    
    reclat = np.linalg.inv( phonon.get_unitcell().get_cell())*2*np.pi# in Angstrom^-1
    qpoints = phonon.get_mesh_dict()['qpoints']
    #Nmodes = len(Vec_freqs)
    Ns = phonon.get_unitcell().get_number_of_atoms()*3
    #Nq = int(Nmodes/Ns)


    C_wtauN = Vec_cqs/(Vec_freqs*Vec_tauN_qs*2*np.pi+eps)# EV/Angstrom**3/K 
    C_wtauN[C_wtauN>THz] = 0.0
    #C_wtauR = Vec_cqs/(Vec_freqs*Vec_tauR_qs*2*np.pi+eps)
    CT0_w2tauN = C_wtauN*T0/(Vec_freqs*2*np.pi+eps) #CT0/(omega^2*tauN) of each mode, in EV/Angstrom**3/THz
    CT0_w2tauN[CT0_w2tauN>THz] = 0.0
    C_tauN = Vec_cqs/(Vec_tauN_qs+eps) # EV*THz/Angstrom**3/K
    C_tauN[C_tauN>THz] = 0.0
    C_tauR = Vec_cqs/(Vec_tauR_qs+eps) # EV*THz/Angstrom**3 /K 
    C_tauR[C_tauR>THz] = 0.0
    
    C_tau = C_tauN + C_tauR
    
    Nscatt_ratio = (Vec_tau_qs+eps)/(Vec_tauN_qs+eps) # N-scattering rate/scattering rate. dimensionless
    Nscatt_ratio[Nscatt_ratio>THz]= 0.0    
    Q_qs = Phonon_GenRate(Vec_cqs,Vec_freqs) # Q0 = 1 unit temperature rise
    


    #A11 = np.tensordot(C_tauR,(1-X_modes),axes=1) # in eV*THz/Angstrom**3 /K
    
    A11 = np.zeros(mesh_shape,dtype=Xdtype); 
    A12 = np.zeros(mesh_shape,dtype=Xdtype);  
    A22 = np.zeros(mesh_shape,dtype=Xdtype);
    
    ARqvec = np.zeros(mesh_shape,dtype=Xdtype)
    ANqvec = np.zeros(mesh_shape,dtype=Xdtype)
    Asubmat = np.zeros(mesh_shape,dtype=Xdtype)
    
    bTR = np.zeros(mesh_shape,dtype=Xdtype);
    bTN = np.zeros(mesh_shape,dtype=Xdtype);
    
    bqvec = np.zeros(mesh_shape,dtype=Xdtype)
    

    Amat = np.zeros((3,3)+mesh_shape,dtype =Xdtype)
    bvec = np.zeros((3,)+mesh_shape,dtype=Xdtype)
    

    XQtau = np.zeros(mesh_shape,dtype=Xdtype) 
    coeff_GTR = np.zeros(mesh_shape,dtype=Xdtype)
    coeff_GTN = np.zeros(mesh_shape,dtype=Xdtype)
 

    for iq in range(len(qpoints)): #range(Nq):

        qfrac = qpoints[iq]
        
        rots_sitesym = rots_qpoints[iq]
        
        multi_q = len(rots_sitesym)
        
        C_wtauN_at_q = C_wtauN[iq*Ns:(iq+1)*Ns] # c/omega*tauN of each mode
        #C_wtauR_at_q = C_wtauR[iq*Ns:(iq+1)*Ns]
        C_tauN_at_q = C_tauN[iq*Ns:(iq+1)*Ns]
        C_tauR_at_q = C_tauR[iq*Ns:(iq+1)*Ns]
        
        CT0_w2tauN_at_q = CT0_w2tauN[iq*Ns:(iq+1)*Ns]
        
        Qs_at_q = Q_qs[iq*Ns:(iq+1)*Ns]
        
        cs_at_q = Vec_cqs[iq*Ns:(iq+1)*Ns]
        
        taus_at_q = Vec_tau_qs[iq*Ns:(iq+1)*Ns]
        
        Fs_at_q = Vec_Fsc_qs[:,iq*Ns:(iq+1)*Ns]
        omegas_at_q = Vec_freqs[iq*Ns:(iq+1)*Ns]*2*np.pi+eps
        ns_at_q = Qs_at_q/omegas_at_q # phonon number generation rate. 
        ns_at_q[ns_at_q>THz] = 0.0
        
        Nscatt_ratio_q = Nscatt_ratio[iq*Ns:(iq+1)*Ns]
        Nratio_mat = np.broadcast_to(Nscatt_ratio_q,mesh_shape+Nscatt_ratio_q.shape)
        Nratio_mat = np.moveaxis(Nratio_mat,len(mesh_shape),0)
        Rratio_mat = 1 - Nratio_mat
        
        
        #X_modes_at_q = np.zeros((Ns,)+mesh_shape,dtype=Xdtype)
        
        
        for rot in rots_sitesym:
            rot_qfrac = np.dot(rot, qfrac) # dimensionless q
            rot_q = np.dot(reclat,rot_qfrac)
               
            rot_q1D = rot_q[ix] # use this when calculating TTG
            
            rot_qq = rot_q1D**2 #np.tensordot(rot_q2D,rot_q2D,axes=0)
            
            qrepmat = np.broadcast_to(rot_q1D,mesh_shape)
            #qrepmat = np.moveaxis(qrepmat,len(mesh_shape),0)
            #
            r_cart = similarity_transformation(reclat, rot)
            
            rot_Fs = np.dot(r_cart,Fs_at_q) #(3,3) by (3,Ns)
            
            #rot_Fs_mz = np.array((rot_Fs[0],rot_Fs[1],-rot_Fs[2]))
            
            
            X_modes_at_rotqpos,Xq_modes_at_q_pos = X1D_modes_at_q(XIx,OMEGAH,rot_Fs,taus_at_q, ix,Xdtype) 
            X_modes_at_rotqneg,Xq_modes_at_q_neg = X1D_modes_at_q(XIx,OMEGAH,-rot_Fs,taus_at_q, ix,Xdtype) 
            
            X_modes_at_rotq = (X_modes_at_rotqpos+X_modes_at_rotqneg)/2.

            # symmetrized using in-plane isotropy
            # X_modes_at_rotq,Xq_modes_at_q_pos = X1Diso_modes_at_q(XIx, OMEGAH, rot_Fs, taus_at_q,Xdtype=Xdtype) #0.5*(X_modes_at_rotqpos+X_modes_at_rotqneg)  
            # X_modes_at_rotq,Xq_modes_at_q_neg = X1Diso_modes_at_q(-XIx, OMEGAH, rot_Fs, taus_at_q,Xdtype=Xdtype)
            
            # the first return is even.
                       
            
            coeffqR_pos = np.tensordot(C_wtauN_at_q,-Rratio_mat*Xq_modes_at_q_pos,axes=1)
            coeffqR_neg = np.tensordot(C_wtauN_at_q,-Rratio_mat*Xq_modes_at_q_neg,axes=1)
            coeffqN_pos = np.tensordot(C_wtauN_at_q,-Nratio_mat*Xq_modes_at_q_pos,axes=1)
            coeffqN_neg = np.tensordot(C_wtauN_at_q,-Nratio_mat*Xq_modes_at_q_neg,axes=1)
            coeffqR = (coeffqR_pos-coeffqR_neg)/multi_q/2.0
            coeffqN = (coeffqN_pos-coeffqN_neg)/multi_q/2.0            

            
            coeffqq = np.tensordot(CT0_w2tauN_at_q,1-Nratio_mat*X_modes_at_rotq,axes=1)/multi_q #*2
            

            
            ARqvec += coeffqR*qrepmat
            ANqvec += coeffqN*qrepmat
            
            
            # b_coeffq_pos = np.tensordot(ns_at_q,Nratio_mat*X_modes_at_rotqpos,axes=1)
            # b_coeffq_neg = np.tensordot(ns_at_q,Nratio_mat*X_modes_at_rotqneg,axes=1)
            
            # # symmetrized using in-plane isotropy
            b_coeffq_pos = np.tensordot(ns_at_q,Nratio_mat*Xq_modes_at_q_pos,axes=1)
            b_coeffq_neg = np.tensordot(ns_at_q,Nratio_mat*Xq_modes_at_q_neg,axes=1)
            
            
            b_coeffq = (b_coeffq_pos-b_coeffq_neg)/multi_q/2.0
            
            
            
             

            bqvec += b_coeffq*qrepmat
            Asubmat += coeffqq*rot_qq
            
        
        
            XQtau += np.tensordot(Qs_at_q*taus_at_q,X_modes_at_rotq,axes=1)/multi_q
            
            summand_R = np.tensordot(cs_at_q*(1-Nscatt_ratio_q),X_modes_at_rotq,axes=1)/multi_q
            summand_R[np.isnan(summand_R)] = 0.0
            summand_R[summand_R>THz] = 0.0
            summand_N = np.tensordot(cs_at_q*Nscatt_ratio_q,X_modes_at_rotq,axes=1)/multi_q
            summand_N[np.isnan(summand_N)] = 0.0
            summand_N[summand_N>THz] = 0.0
            
            coeff_GTR += summand_R
            coeff_GTN += summand_N
            
 
            A11 += np.tensordot(C_tauR_at_q,1-Rratio_mat*X_modes_at_rotq,axes=1)/multi_q
            A12 -= np.tensordot(C_tauR_at_q,Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
            A22 += np.tensordot(C_tauN_at_q,1-Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
        

        
        
            bTR += np.tensordot(Qs_at_q,Rratio_mat*X_modes_at_rotq,axes=1)/multi_q
            bTN += np.tensordot(Qs_at_q,Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
        




    # coefficient of the linear Ax=b
    Amat[0,0] = A11 + eps
    Amat[0,1] = A12 + eps
    Amat[1,0] = A12 + eps
    Amat[1,1] = A22 + eps
    
    
    Amat[0,2] = ARqvec*T0 + eps
    Amat[1,2] = ANqvec*T0 + eps
    
    
    Amat[2,0] = ARqvec + eps
    Amat[2,1] = ANqvec + eps
    
    Amat[2,2] = Asubmat +eps
    
    

    bvec[0] = bTR
    bvec[1] = bTN
    bvec[2] = bqvec 
    
    x =np.zeros((len(bvec),)+mesh_shape,Xdtype)
    
    
    Nw,Nxir = mesh_shape
    for iw in range(Nw):
        for jr in range(Nxir):
              
                xij = np.linalg.solve(Amat[:,:,iw,jr],bvec[:,iw,jr])
                x[:,iw,jr] = xij
        
                    
            
    GdT_R = x[0] # Green's Function for pseudo temperature for N process
    GdT_N = x[1] # Green's function for pseudo temperature for R process
    Gu = x[2] # Green's Function for drift velocity
    
    GdT = 1/np.sum(C_tau)*(np.sum(C_tauR)*GdT_R + np.sum(C_tauN)*GdT_N) 
    
    GdT_RTA = (bTR+bTN)/(A11+2*A12+A22) # Naive RTA without Normal processes
    
    return GdT,Gu,GdT_RTA

def Solve_cyln_TempRN_udrift(XIr,XIz,OMEGAH,Vec_cqs,Vec_freqs,Vec_Fsc_qs,Vec_tauN_qs,Vec_tauR_qs,Vec_tau_qs,T0,phonon,rots_qpoints,rz_directs=(0,1,2),Xdtype='complex64'):
    # xz_directs specifies in-plane and cross-plane directions.
    # corresponding to the dirftless case.
    eps = 1e-50
    
    ir1 = rz_directs[0]
    ir2 = rz_directs[1]
    iz = rz_directs[2]
    
    Nr = XIr.shape[0]
    Nz = XIz.shape[1]
    Nw = OMEGAH.shape[2]
    
    mesh_shape = (Nr,Nz,Nw)
    
    reclat = np.linalg.inv( phonon.get_unitcell().get_cell())*2*np.pi # in Angstrom^-1
    qpoints = phonon.get_mesh_dict()['qpoints']
    #Nmodes = len(Vec_freqs)
    Ns = phonon.get_unitcell().get_number_of_atoms()*3
    #Nq = int(Nmodes/Ns)


    C_wtauN = Vec_cqs/(Vec_freqs*Vec_tauN_qs*2*np.pi+eps)# EV/Angstrom**3/K 
    #C_wtauR = Vec_cqs/(Vec_freqs*Vec_tauR_qs*2*np.pi+eps)
    CT0_w2tauN = C_wtauN*T0/(Vec_freqs*2*np.pi+eps) #CT0/(omega^2*tauN) of each mode, in EV/Angstrom**3/THz
    C_tauN = Vec_cqs/(Vec_tauN_qs+eps) # EV*THz/Angstrom**3/K
    C_tauN[C_tauN>THz] = 0.0
    C_tauR = Vec_cqs/(Vec_tauR_qs+eps) # EV*THz/Angstrom**3 /K 
    C_tauR[C_tauR>THz] = 0.0
    
    C_tau = C_tauN + C_tauR
    Nscatt_ratio = (Vec_tau_qs)/(Vec_tauN_qs+eps) # N-scattering rate/scattering rate. dimensionless    
    Q_qs = Phonon_GenRate(Vec_cqs,Vec_freqs) # Q0 = 1 unit temperature rise
    

   
    A11 = np.zeros(mesh_shape,dtype=Xdtype); 
    A12 = np.zeros(mesh_shape,dtype=Xdtype);  
    A22 = np.zeros(mesh_shape,dtype=Xdtype);
    
    ARqvec = np.zeros((2,)+mesh_shape,dtype=Xdtype)
    ANqvec = np.zeros((2,)+mesh_shape,dtype=Xdtype)
    Asubmat = np.zeros((2,2)+mesh_shape,dtype=Xdtype)
    
    bTR = np.zeros(mesh_shape,dtype=Xdtype);
    bTN = np.zeros(mesh_shape,dtype=Xdtype);
    
    bqvec = np.zeros((2,)+mesh_shape,dtype=Xdtype)
    

    Amat = np.zeros((4,4)+mesh_shape,dtype =Xdtype)
    bvec = np.zeros((4,)+mesh_shape,dtype=Xdtype)
    

    # XQtau = np.zeros(mesh_shape,dtype=Xdtype) 
    # coeff_GTR = np.zeros(mesh_shape,dtype=Xdtype)
    # coeff_GTN = np.zeros(mesh_shape,dtype=Xdtype)
 

    for iq in range(len(qpoints)): #range(Nq):

        qfrac = qpoints[iq]
        
        rots_sitesym = rots_qpoints[iq]
        
        multi_q = len(rots_sitesym)
        
        C_wtauN_at_q = C_wtauN[iq*Ns:(iq+1)*Ns] # c/omega*tauN of each mode
        #C_wtauR_at_q = C_wtauR[iq*Ns:(iq+1)*Ns]
        C_tauN_at_q = C_tauN[iq*Ns:(iq+1)*Ns]
        C_tauR_at_q = C_tauR[iq*Ns:(iq+1)*Ns]
        
        CT0_w2tauN_at_q = CT0_w2tauN[iq*Ns:(iq+1)*Ns]
        
        Qs_at_q = Q_qs[iq*Ns:(iq+1)*Ns]
        ns_at_q = Qs_at_q/(eps+Vec_freqs[iq*Ns:(iq+1)*Ns]*2*np.pi) # phonon number generation rate. 
        # cs_at_q = Vec_cqs[iq*Ns:(iq+1)*Ns]
        
        taus_at_q = Vec_tau_qs[iq*Ns:(iq+1)*Ns]
        Fs_at_q = Vec_Fsc_qs[:,iq*Ns:(iq+1)*Ns]
        #omegas_at_q = Vec_freqs[iq*Ns:(iq+1)*Ns]*2*np.pi+eps
        
        Nscatt_ratio_q = Nscatt_ratio[iq*Ns:(iq+1)*Ns]
        Nratio_mat = np.broadcast_to(Nscatt_ratio_q,mesh_shape+Nscatt_ratio_q.shape)
        Nratio_mat = np.moveaxis(Nratio_mat,len(mesh_shape),0)
        Rratio_mat = 1 - Nratio_mat
        
        
        #X_modes_at_q = np.zeros((Ns,)+mesh_shape,dtype=Xdtype)
        
        
        for rot in rots_sitesym:
            rot_qfrac = np.dot(rot, qfrac) # dimensionless q
            rot_q = np.dot(reclat,rot_qfrac)
            
            rot_qr = np.sqrt(rot_q[ir1]**2+rot_q[ir2]**2)
            rot_qz = rot_q[iz]
            rot_q2D = np.array([rot_qr,rot_qz])
            
            
            rot_qq = np.tensordot(rot_q2D,rot_q2D,axes=0)
            
            qrepmat = np.broadcast_to(rot_q2D,mesh_shape+rot_q2D.shape)
            qrepmat = np.moveaxis(qrepmat,len(mesh_shape),0)
            
            r_cart = similarity_transformation(reclat, rot)
            
            rot_Fs = np.dot(r_cart,Fs_at_q) #(3,3) by (3,Ns)
            
            #rot_Fs_mz = np.array((rot_Fs[0],rot_Fs[1],-rot_Fs[2]))
            

            
            # X_modes_at_rotq,Xq_modes_at_rotq = Xcyln_modes_at_q(XIr, XIz, OMEGAH, rot_Fs, taus_at_q , rz_directs, Xdtype)
            
            Fr_at_q = np.sqrt(rot_Fs[ir1]**2 + rot_Fs[ir1]**2)
            Fz_at_q = rot_Fs[iz]
            
            X_modes_at_rotq_pos,Xq_modes_at_rotq_pos =  Xcyln_modes_at_q(XIr, XIz, OMEGAH, Fr_at_q, Fz_at_q, taus_at_q, Xdtype) #Xcyln_modes_at_q(XIr, XIz, OMEGAH, rot_Fs, taus_at_q , rz_directs, Xdtype)
            X_modes_at_rotq_neg,Xq_modes_at_rotq_neg = Xcyln_modes_at_q(XIr, XIz, OMEGAH, -Fr_at_q, -Fz_at_q, taus_at_q, Xdtype)
            
            X_modes_at_rotq = 0.5*(X_modes_at_rotq_pos+X_modes_at_rotq_neg)


            # coeffqR = -np.tensordot(C_wtauN_at_q,Rratio_mat*Xq_modes_at_rotq,axes=1)/multi_q
            # coeffqN = np.tensordot(C_wtauN_at_q,-Nratio_mat*Xq_modes_at_rotq,axes=1)/multi_q
                       
            coeffqR_pos =  -np.tensordot(C_wtauN_at_q,Rratio_mat*Xq_modes_at_rotq_pos,axes=1)
            coeffqR_neg =  -np.tensordot(C_wtauN_at_q,Rratio_mat*Xq_modes_at_rotq_neg,axes=1)
            coeffqN_pos =  np.tensordot(C_wtauN_at_q, -Nratio_mat*Xq_modes_at_rotq_pos,axes=1)
            coeffqN_neg =  np.tensordot(C_wtauN_at_q,-Nratio_mat*Xq_modes_at_rotq_neg,axes=1)
            
            coeffqR = (coeffqR_pos-coeffqR_neg)/multi_q/2.0 
            coeffqN = (coeffqN_pos-coeffqN_neg)/multi_q/2.0
            
            coeffqq = np.tensordot(CT0_w2tauN_at_q,1-Nratio_mat*X_modes_at_rotq,axes=1)
            

            
            ARqvec += coeffqR*qrepmat
            ANqvec += coeffqN*qrepmat
            
            # b_coeffq = np.tensordot(ns_at_q,Nratio_mat*Xq_modes_at_rotq,axes=1)/multi_q
            
            
            b_coeffq_pos = np.tensordot(ns_at_q,Nratio_mat*Xq_modes_at_rotq_pos,axes=1)
            b_coeffq_neg = np.tensordot(ns_at_q,Nratio_mat*Xq_modes_at_rotq_neg,axes=1)
            b_coeffq = 0.5*(b_coeffq_pos-b_coeffq_neg)/multi_q 
            
            
            

            bqvec += b_coeffq*qrepmat
            
            for (i,j) in ((0,0),(0,1),(1,0),(1,1)): #((0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)):
                Asubmat[i,j] += coeffqq*rot_qq[i,j]/multi_q
        
        
            # XQtau += np.tensordot(Qs_at_q*taus_at_q,X_modes_at_rotq,axes=1)/multi_q
            # coeff_GTR += np.tensordot(cs_at_q*(1-Nscatt_ratio_q),X_modes_at_rotq,axes=1)/multi_q
            # coeff_GTN += np.tensordot(cs_at_q*Nscatt_ratio_q,X_modes_at_rotq,axes=1)/multi_q
            
     
        
            A11 += np.tensordot(C_tauR_at_q,1-Rratio_mat*X_modes_at_rotq,axes=1)/multi_q
            A12 -= np.tensordot(C_tauR_at_q,Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
            A22 += np.tensordot(C_tauN_at_q,1-Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
             
        
            bTR += np.tensordot(Qs_at_q,Rratio_mat*X_modes_at_rotq,axes=1)/multi_q
            bTN += np.tensordot(Qs_at_q,Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
        

    # coefficient of the linear Ax=b
    Amat[0,0] = A11 + eps
    Amat[0,1] = A12 + eps
    Amat[1,0] = A12 + eps
    Amat[1,1] = A22 + eps
    
    
    Amat[0,2:] = ARqvec*T0 + eps
    Amat[1,2:] = ANqvec*T0 + eps
    
    
    Amat[2:,0] = ARqvec + eps
    Amat[2:,1] = ANqvec + eps
    
    Amat[2:,2:] = Asubmat +eps
    
    

    bvec[0] = bTR
    bvec[1] = bTN
    bvec[2:] = bqvec 
    
    x =np.zeros((len(bvec),)+mesh_shape,Xdtype)
    
    
    Nw,Nxir,Nxiz = mesh_shape
    for iw in range(Nw):
        for jr in range(Nxir):
              for kz in range(Nxiz):
                    xijk = np.linalg.solve(Amat[:,:,iw,jr,kz],bvec[:,iw,jr,kz])
                    x[:,iw,jr,kz] = xijk
        
                    
            
    GdT_R = x[0] # Green's Function for pseudo temperature for N process
    GdT_N = x[1] # Green's function for pseudo temperature for R process
    Gu = x[2:] # Green's Function for drift velocity
    
    
    GdT = 1/np.sum(C_tau)*(np.sum(C_tauR)*GdT_R + np.sum(C_tauN)*GdT_N) 
    
    GdT_RTA = (bTR+bTN)/(A11+2*A12+A22) # Naive RTA without Normal processes
    
    return GdT,Gu,GdT_RTA


def Solve2D_TempRN_udrift(XIx,XIy,OMEGAH,Vec_cqs,Vec_freqs,Vec_Fsc_qs,Vec_tauN_qs,Vec_tauR_qs,Vec_tau_qs,T0,phonon,rots_qpoints,xy_directs=(0,1),Xdtype='complex64'):
    # xz_directs specifies in-plane and cross-plane directions.
    eps = 1e-50
    
    ix = xy_directs[0]
    iy = xy_directs[1]
    
    Nx = XIx.shape[0]
    Ny = XIy.shape[1]
    Nw = OMEGAH.shape[2]
    
    mesh_shape = (Nx,Ny,Nw)
    
    reclat = np.linalg.inv( phonon.get_unitcell().get_cell())*2*np.pi # in Angstrom^-1
    qpoints = phonon.get_mesh_dict()['qpoints']
    #Nmodes = len(Vec_freqs)
    Ns = phonon.get_unitcell().get_number_of_atoms()*3
    #Nq = int(Nmodes/Ns)


    C_wtauN = Vec_cqs/(Vec_freqs*Vec_tauN_qs*2*np.pi+eps)# EV/Angstrom**3/K 
    C_wtauN[C_wtauN>THz] = 0.0
    #C_wtauR = Vec_cqs/(Vec_freqs*Vec_tauR_qs*2*np.pi+eps)
    CT0_w2tauN = C_wtauN*T0/(Vec_freqs*2*np.pi+eps) #CT0/(omega^2*tauN) of each mode, in EV/Angstrom**3/THz
    CT0_w2tauN[CT0_w2tauN>THz] = 0.0
    C_tauN = Vec_cqs/(Vec_tauN_qs+eps) # EV*THz/Angstrom**3/K
    C_tauN[C_tauN>THz] = 0.0
    C_tauR = Vec_cqs/(Vec_tauR_qs+eps) # EV*THz/Angstrom**3 /K 
    C_tauR[C_tauR>THz] = 0.0
    
    C_tau = C_tauR+C_tauN
    
    C_tau = Vec_cqs/(Vec_tau_qs+eps)
    C_tau[C_tau>THz] = 0.0
    
    Nscatt_ratio = (Vec_tau_qs)/(Vec_tauN_qs+eps) # N-scattering rate/scattering rate. dimensionless    
    Q_qs = Phonon_GenRate(Vec_cqs,Vec_freqs) # Q0 = 1 unit temperature rise
    
    # XQtau = np.zeros(mesh_shape,dtype=Xdtype) 
    # coeff_GTR = np.zeros(mesh_shape,dtype=Xdtype)
    # coeff_GTN = np.zeros(mesh_shape,dtype=Xdtype)

    #A11 = np.tensordot(C_tauR,(1-X_modes),axes=1) # in eV*THz/Angstrom**3 /K
    
    A11 = np.zeros(mesh_shape,dtype=Xdtype); 
    A12 = np.zeros(mesh_shape,dtype=Xdtype);  
    A22 = np.zeros(mesh_shape,dtype=Xdtype);
    
    ARqvec = np.zeros((2,)+mesh_shape,dtype=Xdtype)
    ANqvec = np.zeros((2,)+mesh_shape,dtype=Xdtype)
    Asubmat = np.zeros((2,2)+mesh_shape,dtype=Xdtype)
    
    bTR = np.zeros(mesh_shape,dtype=Xdtype);
    bTN = np.zeros(mesh_shape,dtype=Xdtype);
    
    bqvec = np.zeros((2,)+mesh_shape,dtype=Xdtype)
    

    Amat = np.zeros((4,4)+mesh_shape,dtype =Xdtype)
    bvec = np.zeros((4,)+mesh_shape,dtype=Xdtype)
    
 

    for iq in range(len(qpoints)): #range(Nq):
        
        #qfrac = qpoints[iq]


        qfrac = qpoints[iq]
        
        rots_sitesym = rots_qpoints[iq]
        
        multi_q = len(rots_sitesym)
        
        C_wtauN_at_q = C_wtauN[iq*Ns:(iq+1)*Ns] # c/omega*tauN of each mode
        #C_wtauR_at_q = C_wtauR[iq*Ns:(iq+1)*Ns]
        C_tauN_at_q = C_tauN[iq*Ns:(iq+1)*Ns]
        C_tauR_at_q = C_tauR[iq*Ns:(iq+1)*Ns]
        
        CT0_w2tauN_at_q = CT0_w2tauN[iq*Ns:(iq+1)*Ns]
        
        Qs_at_q = Q_qs[iq*Ns:(iq+1)*Ns]
        ns_at_q = Qs_at_q/(eps+Vec_freqs[iq*Ns:(iq+1)*Ns]*2*np.pi) # phonon number generation rate. 
        
        # cs_at_q = Vec_cqs[iq*Ns:(iq+1)*Ns]
        
        taus_at_q = Vec_tau_qs[iq*Ns:(iq+1)*Ns]
        Fs_at_q = Vec_Fsc_qs[:,iq*Ns:(iq+1)*Ns]
        # vs_at_q = Vec_vqs[:,iq*Ns:(iq+1)*Ns]
        # omegas_at_q = Vec_freqs[iq*Ns:(iq+1)*Ns]*2*np.pi+eps
        
        Nscatt_ratio_q = Nscatt_ratio[iq*Ns:(iq+1)*Ns]
        Nratio_mat = np.broadcast_to(Nscatt_ratio_q,mesh_shape+Nscatt_ratio_q.shape)
        Nratio_mat = np.moveaxis(Nratio_mat,len(mesh_shape),0)
        Rratio_mat = 1 - Nratio_mat
        
        
        #X_modes_at_q = np.zeros((Ns,)+mesh_shape,dtype=Xdtype)
        
        
        for rot in rots_sitesym:
            rot_qfrac = np.dot(rot, qfrac) # dimensionless q
            rot_q = np.dot(reclat,rot_qfrac)
            
            rot_q2D = np.array((rot_q[ix],rot_q[iy]))
            
            rot_qq = np.tensordot(rot_q2D,rot_q2D,axes=0)
            
            qrepmat = np.broadcast_to(rot_q2D,mesh_shape+rot_q2D.shape)
            qrepmat = np.moveaxis(qrepmat,len(mesh_shape),0) # qdim x mesh_shape
            
            
            
            r_cart = similarity_transformation(reclat, rot)
            
            rot_Fs = np.dot(r_cart,Fs_at_q) #(3,3) by (3,Ns)
            

            X_modes_at_rotq, Xq_modes_at_rotq = X2Dxy_modes_at_q(XIx, XIy, OMEGAH, rot_Fs, taus_at_q,xy_directs,Xdtype)

            coeffqR = -np.tensordot(C_wtauN_at_q,Rratio_mat*Xq_modes_at_rotq,axes=1)/multi_q
            coeffqN = np.tensordot(C_wtauN_at_q,-Nratio_mat*Xq_modes_at_rotq,axes=1)/multi_q

            coeffqq = np.tensordot(CT0_w2tauN_at_q,1-Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
            
            ARqvec += coeffqR*qrepmat            
            ANqvec += coeffqN*qrepmat
            
            
            b_coeffq = np.tensordot(ns_at_q,Nratio_mat*Xq_modes_at_rotq,axes=1)/multi_q
            
            bqvec += b_coeffq*qrepmat[0]
            
            for (i,j) in ((0,0),(0,1),(1,0),(1,1)):
                Asubmat[i,j] += coeffqq*rot_qq[i,j]
                
            # XQtau += np.tensordot(Qs_at_q*taus_at_q,X_modes_at_rotq,axes=1)/multi_q
            # coeff_GTR += np.tensordot(cs_at_q*(1-Nscatt_ratio_q),X_modes_at_rotq,axes=1)/multi_q
            # coeff_GTN += np.tensordot(cs_at_q*Nscatt_ratio_q,X_modes_at_rotq,axes=1)/multi_q
        

        
            A11 += np.tensordot(C_tauR_at_q,1-Rratio_mat*X_modes_at_rotq,axes=1)/multi_q
            A12 -= np.tensordot(C_tauR_at_q,Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
            A22 += np.tensordot(C_tauN_at_q,1-Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
          
            bTR += np.tensordot(Qs_at_q,Rratio_mat*X_modes_at_rotq,axes=1)/multi_q
            bTN += np.tensordot(Qs_at_q,Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
        




    # coefficient of the linear Ax=b
    Amat[0,0] = A11 + eps
    Amat[0,1] = A12 + eps
    Amat[1,0] = A12 + eps
    Amat[1,1] = A22 + eps
    
    
    Amat[0,2:] = ARqvec*T0 + eps
    Amat[1,2:] = ANqvec*T0 + eps
    
    
    Amat[2:,0] = ARqvec + eps
    Amat[2:,1] = ANqvec + eps
    
    Amat[2:,2:] = Asubmat +eps
    
    

    bvec[0] = bTR
    bvec[1] = bTN
    bvec[2:] = bqvec 
    
    x =np.zeros((len(bvec),)+mesh_shape,Xdtype)
    
    
    Nw,Nxix,Nxiy = mesh_shape
    for iw in range(Nw):
        for jx in range(Nxix):
              for ky in range(Nxiy):
                    xijk = np.linalg.solve(Amat[:,:,iw,jx,ky],bvec[:,iw,jx,ky])
                    x[:,iw,jx,ky] = xijk
        
                    
            
    GdT_R = x[0] # Green's Function for pseudo temperature for N process
    GdT_N = x[1] # Green's function for pseudo temperature for R process
    Gu = x[2:] # Green's Function for drift velocity
    
    
    GdT = 1/np.sum(C_tau)*(np.sum(C_tauR)*GdT_R + np.sum(C_tauN)*GdT_N) 
    
    GdT_RTA = (bTR+bTN)/(A11+2*A12+A22) # Naive RTA with out Normal processes
    
    return GdT,Gu,GdT_RTA


def Solve3D_TempRN_udrift(XIx,XIy,XIz,OMEGAH,Vec_cqs,Vec_freqs,Vec_Fsc_qs,Vec_tauN_qs,Vec_tauR_qs,Vec_tau_qs,T0,phonon,rots_qpoints,Xdtype='complex64'):
    # xz_directs specifies in-plane and cross-plane directions.
    eps = 1e-50
    
    
    Nx = XIx.shape[0]
    Ny = XIy.shape[1]
    Nz = XIz.shape[2]
    Nw = OMEGAH.shape[3]
    
    mesh_shape = (Nx,Ny,Nz,Nw)
    
    reclat = np.linalg.inv( phonon.get_unitcell().get_cell())*2*np.pi # in Angstrom^-1
    qpoints = phonon.get_mesh_dict()['qpoints']
    #Nmodes = len(Vec_freqs)
    Ns = phonon.get_unitcell().get_number_of_atoms()*3
    #Nq = int(Nmodes/Ns)


    C_wtauN = Vec_cqs/(Vec_freqs*Vec_tauN_qs*2*np.pi+eps)# EV/Angstrom**3/K 
    #C_wtauR = Vec_cqs/(Vec_freqs*Vec_tauR_qs*2*np.pi+eps)
    CT0_w2tauN = C_wtauN*T0/(Vec_freqs*2*np.pi+eps) #CT0/(omega^2*tauN) of each mode, in EV/Angstrom**3/THz
    C_tauN = Vec_cqs/(Vec_tauN_qs+eps) # EV*THz/Angstrom**3/K
    C_tauN[C_tauN>THz] = 0.0
    C_tauR = Vec_cqs/(Vec_tauR_qs+eps) # EV*THz/Angstrom**3 /K 
    C_tauR[C_tauR>THz] = 0.0
    
    C_tau = C_tauR + C_tauN
    
    Nscatt_ratio = (Vec_tau_qs)/(Vec_tauN_qs+eps) # N-scattering rate/scattering rate. dimensionless    
    Q_qs = Phonon_GenRate(Vec_cqs,Vec_freqs) # Q0 = 1 unit temperature rise
    


    #A11 = np.tensordot(C_tauR,(1-X_modes),axes=1) # in eV*THz/Angstrom**3 /K
    
    A11 = np.zeros(mesh_shape,dtype=Xdtype); 
    A12 = np.zeros(mesh_shape,dtype=Xdtype);  
    A22 = np.zeros(mesh_shape,dtype=Xdtype);
    
    ARqvec = np.zeros((3,)+mesh_shape,dtype=Xdtype)
    ANqvec = np.zeros((3,)+mesh_shape,dtype=Xdtype)
    Asubmat = np.zeros((3,3)+mesh_shape,dtype=Xdtype)
    
    bTR = np.zeros(mesh_shape,dtype=Xdtype);
    bTN = np.zeros(mesh_shape,dtype=Xdtype);
    
    bqvec = np.zeros((3,)+mesh_shape,dtype=Xdtype)
    

    Amat = np.zeros((5,5)+mesh_shape,dtype =Xdtype)
    bvec = np.zeros((5,)+mesh_shape,dtype=Xdtype)
    

    # XQtau = np.zeros(mesh_shape,dtype=Xdtype) 
    # coeff_GTR = np.zeros(mesh_shape,dtype=Xdtype)
    # coeff_GTN = np.zeros(mesh_shape,dtype=Xdtype)
 

    for iq in range(len(qpoints)): #range(Nq):
        
        #qfrac = qpoints[iq]


        qfrac = qpoints[iq]
        
        rots_sitesym = rots_qpoints[iq]
        
        multi_q = len(rots_sitesym)
        
        C_wtauN_at_q = C_wtauN[iq*Ns:(iq+1)*Ns] # c/omega*tauN of each mode
        #C_wtauR_at_q = C_wtauR[iq*Ns:(iq+1)*Ns]
        C_tauN_at_q = C_tauN[iq*Ns:(iq+1)*Ns]
        C_tauR_at_q = C_tauR[iq*Ns:(iq+1)*Ns]
        
        CT0_w2tauN_at_q = CT0_w2tauN[iq*Ns:(iq+1)*Ns]
        
        Qs_at_q = Q_qs[iq*Ns:(iq+1)*Ns]
        ns_at_q = Qs_at_q/(eps+Vec_freqs[iq*Ns:(iq+1)*Ns]*2*np.pi) # phonon number generation rate. 
        # cs_at_q = Vec_cqs[iq*Ns:(iq+1)*Ns]
        
        taus_at_q = Vec_tau_qs[iq*Ns:(iq+1)*Ns]
        Fs_at_q = Vec_Fsc_qs[:,iq*Ns:(iq+1)*Ns]
        # omegas_at_q = Vec_freqs[iq*Ns:(iq+1)*Ns]*2*np.pi+eps
        
        Nscatt_ratio_q = Nscatt_ratio[iq*Ns:(iq+1)*Ns]
        Nratio_mat = np.broadcast_to(Nscatt_ratio_q,mesh_shape+Nscatt_ratio_q.shape)
        Nratio_mat = np.moveaxis(Nratio_mat,len(mesh_shape),0)
        Rratio_mat = 1 - Nratio_mat
        
        
        #X_modes_at_q = np.zeros((Ns,)+mesh_shape,dtype=Xdtype)
        
        
        for rot in rots_sitesym:
            rot_qfrac = np.dot(rot, qfrac) # dimensionless q
            rot_q = np.dot(reclat,rot_qfrac)
            
            
            
            rot_qq = np.tensordot(rot_q,rot_q,axes=0)
            
            qrepmat = np.broadcast_to(rot_q,mesh_shape+rot_q.shape)
            qrepmat = np.moveaxis(qrepmat,len(mesh_shape),0)
            
            r_cart = similarity_transformation(reclat, rot)
            
            rot_Fs = np.dot(r_cart,Fs_at_q) #(3,3) by (3,Ns)
            
            #rot_Fs_mz = np.array((rot_Fs[0],rot_Fs[1],-rot_Fs[2]))
            
            X_modes_at_rotqpos = X3D_modes_at_q(XIx,XIy,XIz,OMEGAH,rot_Fs,taus_at_q,Xdtype)
            X_modes_at_rotqneg = X3D_modes_at_q(XIx,XIy,XIz,OMEGAH,-rot_Fs,taus_at_q,Xdtype)
            X_modes_at_rotq = 0.5* (X_modes_at_rotqpos+X_modes_at_rotqneg)
            
            
            
            
            # coeffqR =  -np.tensordot(C_wtauN_at_q,Rratio_mat*X_modes_at_rotq,axes=1)/multi_q
            # coeffqN =  np.tensordot(C_wtauN_at_q,1-Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
            
            coeffqR_pos =  -np.tensordot(C_wtauN_at_q,Rratio_mat*X_modes_at_rotqpos,axes=1)
            coeffqR_neg =  -np.tensordot(C_wtauN_at_q,Rratio_mat*X_modes_at_rotqneg,axes=1)
            coeffqN_pos =  np.tensordot(C_wtauN_at_q,1-Nratio_mat*X_modes_at_rotqpos,axes=1)
            coeffqN_neg =  np.tensordot(C_wtauN_at_q,1-Nratio_mat*X_modes_at_rotqneg,axes=1)
            coeffqR = (coeffqR_pos-coeffqR_neg)/multi_q/2.0 
            coeffqN = (coeffqN_pos-coeffqN_neg)/multi_q/2.0
            
            coeffqq = np.tensordot(CT0_w2tauN_at_q,1-Nratio_mat*X_modes_at_rotq,axes=1)
            
            
            ARqvec += coeffqR*qrepmat
            ANqvec += coeffqN*qrepmat
            
            # b_coeffq = np.tensordot(ns_at_q,Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
            
            b_coeffq_pos = np.tensordot(ns_at_q,Nratio_mat*X_modes_at_rotqpos,axes=1)
            b_coeffq_neg = np.tensordot(ns_at_q,Nratio_mat*X_modes_at_rotqneg,axes=1)
            b_coeffq = 0.5*(b_coeffq_pos-b_coeffq_neg)/multi_q 
            

            bqvec += b_coeffq*qrepmat
            
            for (i,j) in ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)):
                Asubmat[i,j] += coeffqq*rot_qq[i,j]/multi_q
        
        
            #XQtau += np.tensordot(Qs_at_q*taus_at_q,X_modes_at_rotq,axes=1)/multi_q
            #coeff_GTR += np.tensordot(cs_at_q*(1-Nscatt_ratio_q),X_modes_at_rotq,axes=1)/multi_q
            #coeff_GTN += np.tensordot(cs_at_q*Nscatt_ratio_q,X_modes_at_rotq,axes=1)/multi_q
            
     


        
            A11 += np.tensordot(C_tauR_at_q,1-Rratio_mat*X_modes_at_rotq,axes=1)/multi_q
            A12 -= np.tensordot(C_tauR_at_q,Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
            A22 += np.tensordot(C_tauN_at_q,1-Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
        

        
        
            bTR += np.tensordot(Qs_at_q,Rratio_mat*X_modes_at_rotq,axes=1)/multi_q
            bTN += np.tensordot(Qs_at_q,Nratio_mat*X_modes_at_rotq,axes=1)/multi_q
        




    # coefficient of the linear Ax=b
    Amat[0,0] = A11 + eps
    Amat[0,1] = A12 + eps
    Amat[1,0] = A12 + eps
    Amat[1,1] = A22 + eps
    
    
    Amat[0,2:] = ARqvec*T0 + eps
    Amat[1,2:] = ANqvec*T0 + eps
    
    
    Amat[2:,0] = ARqvec + eps
    Amat[2:,1] = ANqvec + eps
    
    Amat[2:,2:] = Asubmat +eps
    
    

    bvec[0] = bTR
    bvec[1] = bTN
    bvec[2:] = bqvec 
    
    x =np.zeros((len(bvec),)+mesh_shape,Xdtype)
    
    
    Nw,Nxix,Nxiy,Nxiz = mesh_shape
    for iw in range(Nw):
        for jx in range(Nxix):
            for ky in range(Nxiy):   
                for lz in range(Nxiz):
                    xijkl = np.linalg.solve(Amat[:,:,iw,jx,ky,lz],bvec[:,iw,jx,ky,lz])
                    x[:,iw,jx,ky,lz] = xijkl
        
                    
            
    GdT_R = x[0] # Green's Function for pseudo temperature for N process
    GdT_N = x[1] # Green's function for pseudo temperature for R process
    Gu = x[2:] # Green's Function for drift velocity

    
    GdT = 1/np.sum(C_tau)*(np.sum(C_tauR)*GdT_R + np.sum(C_tauN)*GdT_N) 
    
    GdT_RTA = (bTR+bTN)/(A11+2*A12+A22) # Naive RTA without Normal processes
    
    return GdT,Gu,GdT_RTA 

def get_cyln_BTEGFs(T0,load_GFs,is_isotope,XIr,XIz,OMEGAH,qpoints_full,phonon,rots_qpoints, 
                    Vec_freqs,Vec_cqs,Vec_vqs,Vec_Fsc_qs,Vec_tau_qs,Vec_tauN_qs,Vec_tauR_qs,directions=(0,1,2),Xdtype='complex64'):
    
    mesh =  phonon._mesh.get_mesh_numbers()
    
    
    if load_GFs == False:
        GdT_NU,Gu,GdT_RTA = Solve_cyln_TempRN_udrift(XIr, XIz, OMEGAH, Vec_cqs, Vec_freqs, Vec_vqs*Vec_tau_qs*Angstrom, Vec_tauN_qs, Vec_tauR_qs, Vec_tau_qs, T0, phonon, rots_qpoints, directions, Xdtype)
        if is_isotope:
            GF_BTE_file = 'GF_cyln_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'.hdf5'
        else:
            GF_BTE_file = 'GF_cyln_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'-noiso.hdf5'
        GF_h5 = h5py.File(GF_BTE_file,'w')
        GF_h5.create_dataset('XIr',data=XIr)
        GF_h5.create_dataset('XIz',data=XIz)
        GF_h5.create_dataset('OMEGAH',data=OMEGAH)
        GF_h5.create_dataset('GdT_NU',data=GdT_NU)
        GF_h5.create_dataset('Gu',data=Gu)
        GF_h5.create_dataset('GdT_RTA',data=GdT_RTA)
            #MeshGrid = (XIr,XIz,OMEGAH)
    
    else:
        if is_isotope:
            GF_BTE_file = 'GF_cyln_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'.hdf5'
        else:
            GF_BTE_file = 'GF_cyln_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'-noiso.hdf5'
        GF_h5 = h5py.File(GF_BTE_file,'r')
        XIr = np.array(GF_h5.get('XIr'))
        XIz = np.array(GF_h5.get('XIz'))
        OMEGAH = np.array(GF_h5.get('OMEGAH'))
        GdT_NU = np.array(GF_h5.get('GdT_NU'))
        Gu = np.array(GF_h5.get('Gu'))
        GdT_RTA = np.array(GF_h5.get('GdT_RTA'))
        GF_h5.close()
        #MeshGrid = (XIr,XIz,OMEGAH)
            
            
    return XIr,XIz,OMEGAH,GdT_NU,Gu,GdT_RTA


def get_BTEGFs(T0,load_GFs,is_isotope,MeshGrid_in,qpoints_full,phonon,rots_qpoints,
               Vec_freqs,Vec_cqs,Vec_vqs,Vec_Fsc_qs,Vec_tau_qs,Vec_tauN_qs,Vec_tauR_qs,directions,Xdtype='complex64'):
    
    # compute susceptibility first
    Dim = len(MeshGrid_in)-1 #space Dimension
    
    MeshGrid = MeshGrid_in
    
    if Dim == 1:
        XIx, OMEGAH = MeshGrid_in 
        if len(directions)>1:
            print('1D calculation, take the first specified direction')
            x_direct = directions[0]
        else:
            x_direct = directions
    elif Dim == 2:    
        XIx,XIy,OMEGAH = MeshGrid_in
        if len(directions)>2:
            print('2D calculation, take the two specified direction')
            xy_directs = (directions[0],directions[1])
        else:
            xy_directs = directions
    elif Dim == 3:
        XIx,XIy,XIz,OMEGAH = MeshGrid_in
        
    else:
        print('Only supporting 1D 2D and 3D solutions')
        return None
    
    
    mesh =  phonon._mesh.get_mesh_numbers()

            
    if load_GFs == False:
        if Dim==1:
            GdT_NU,Gu,GdT_RTA = Solve1D_TempRN_udrift(XIx, OMEGAH, Vec_cqs, Vec_freqs, Vec_Fsc_qs, Vec_tauN_qs, Vec_tauR_qs, Vec_tau_qs, T0, phonon, rots_qpoints,x_direct,Xdtype)
            if is_isotope:
                GF_BTE_file = 'GF1D_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'.hdf5'
            else:
                GF_BTE_file = 'GF1D_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'-noiso.hdf5'
            GF_h5 = h5py.File(GF_BTE_file,'w')
            GF_h5.create_dataset('XIx',data=XIx)
            GF_h5.create_dataset('OMEGAH',data=OMEGAH)
            GF_h5.create_dataset('GdT_NU',data=GdT_NU)
            GF_h5.create_dataset('Gu',data=Gu)
            GF_h5.create_dataset('GdT_RTA',data=GdT_RTA)
            GF_h5.close()
        
        if Dim ==2:
        
            GdT_NU,Gu,GdT_RTA = Solve2D_TempRN_udrift(XIx,XIy,OMEGAH,Vec_cqs,Vec_freqs,Vec_Fsc_qs,Vec_tauN_qs,Vec_tauR_qs,Vec_tau_qs,T0,phonon,rots_qpoints,xy_directs,Xdtype)
            if is_isotope:
                GF_BTE_file = 'GF2D_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'.hdf5'
            else:
                GF_BTE_file = 'GF2D_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'-noiso.hdf5'
            GF_h5 = h5py.File(GF_BTE_file,'w')
            GF_h5.create_dataset('XIx',data=XIx)
            GF_h5.create_dataset('XIy',data=XIy)
            GF_h5.create_dataset('OMEGAH',data=OMEGAH)
            GF_h5.create_dataset('GdT_NU',data=GdT_NU)
            GF_h5.create_dataset('Gu',data=Gu)
            GF_h5.create_dataset('GdT_RTA',data=GdT_RTA)
            GF_h5.close()
            
        if Dim ==3:
        
            GdT_NU,Gu,GdT_RTA = Solve3D_TempRN_udrift(XIx, XIy, XIz, OMEGAH, Vec_cqs, Vec_freqs, Vec_Fsc_qs, Vec_tauN_qs, Vec_tauR_qs, Vec_tau_qs, T0, phonon, rots_qpoints, Xdtype)
            if is_isotope:
                GF_BTE_file = 'GF3D_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'.hdf5'
            else:
                GF_BTE_file = 'GF3D_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'-noiso.hdf5'
            GF_h5 = h5py.File(GF_BTE_file,'w')
            GF_h5.create_dataset('XIx',data=XIx)
            GF_h5.create_dataset('XIy',data=XIy)
            GF_h5.create_dataset('XIz',data=XIz)
            GF_h5.create_dataset('OMEGAH',data=OMEGAH)
            GF_h5.create_dataset('GdT_NU',data=GdT_NU)
            GF_h5.create_dataset('Gu',data=Gu)
            GF_h5.create_dataset('GdT_RTA',data=GdT_RTA)
            GF_h5.close()
    else:
        if Dim==1:
            if is_isotope:
                GF_BTE_file = 'GF1D_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'.hdf5'
            else:
                GF_BTE_file = 'GF1D_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'-noiso.hdf5'
    
            GF_h5 = h5py.File(GF_BTE_file,'r')
            XIx = np.array(GF_h5.get('XIx'))
            OMEGAH = np.array(GF_h5.get('OMEGAH'))
            GdT_NU = np.array(GF_h5.get('GdT_NU'))
            Gu = np.array(GF_h5.get('Gu'))
            GdT_RTA = np.array(GF_h5.get('GdT_RTA')) 
            GF_h5.close()
            MeshGrid = (XIx,OMEGAH)
        
        if Dim==2:
            if is_isotope:
                GF_BTE_file = 'GF2D_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'.hdf5'
            else:
                GF_BTE_file = 'GF2D_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'-noiso.hdf5'
    
            GF_h5 = h5py.File(GF_BTE_file,'r')
            XIx = np.array(GF_h5.get('XIx'))
            XIy = np.array(GF_h5.get('XIy'))
            OMEGAH = np.array(GF_h5.get('OMEGAH'))
            GdT_NU = np.array(GF_h5.get('GdT_NU'))
            Gu = np.array(GF_h5.get('Gu'))
            GdT_RTA = np.array(GF_h5.get('GdT_RTA'))
            GF_h5.close()
            MeshGrid = (XIx,XIy,OMEGAH)
        if Dim==3:
            if is_isotope:
                GF_BTE_file = 'GF3D_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'.hdf5'
            else:
                GF_BTE_file = 'GF3D_NU-RTA_T'+str(T0)+'K_Qmesh'+str(mesh[0])+str(mesh[1])+str(mesh[2])+'-noiso.hdf5'
    
            GF_h5 = h5py.File(GF_BTE_file,'r')
            XIx = np.array(GF_h5.get('XIx'))
            XIy = np.array(GF_h5.get('XIy'))
            XIz = np.array(GF_h5.get('XIz'))
            OMEGAH = np.array(GF_h5.get('OMEGAH'))
            GdT_NU = np.array(GF_h5.get('GdT_NU'))
            Gu = np.array(GF_h5.get('Gu'))
            GdT_RTA = np.array(GF_h5.get('GdT_RTA'))
            GF_h5.close()
            MeshGrid = (XIx,XIy,XIz,OMEGAH)       
    return MeshGrid,GdT_NU,Gu,GdT_RTA


# ----------------------------------- Use Callaway model to compute VHE coeffs ------------------------------------#
def calc_SED_secsound(k,omega_plus):
    
    eps =  np.min([0.0001,np.min(np.abs(omega_plus.imag))*0.02])
    omega_real = omega_plus.real
    omega_imag = omega_plus.imag  + eps
    
    # interpolate
    kvec = np.linspace(np.min(k),np.max(k),401)
    omega_intp = np.interp(kvec,k,omega_real)
    gamma_intp = np.interp(kvec,k,omega_imag)
    
    # omega range
    omega_vec = np.linspace(0,np.max(omega_intp)*2,401)
    
    SPECFUNC = np.zeros([len(kvec),len(omega_vec)])
    
    for i,ki in enumerate(kvec):
        w0 = omega_intp[i]
        g0 = gamma_intp[i]
        
        SPECFUNC[i] = 0.5*(np.abs(g0)+eps)/((omega_vec-w0)**2+g0**2/4+eps)
        
    K,OMEGA = np.meshgrid(kvec,omega_vec,indexing='ij')
    return K,OMEGA,SPECFUNC 
    
    

def get_secsound_dispersion(k,alpha,eta,v0,gamma,lowOrder=True):
    
    B = (alpha+eta)*k**2+gamma
    vss2 = v0**2+gamma/2*(alpha-eta)
    
    vss = np.sqrt(vss2)
    
    vss[np.isnan(vss)] = 0 # smaller than zero
        
    if lowOrder:
        D = (vss*k)**2 - gamma**2/4 + 0j
    else:
        D = (v0*k)**2-(gamma+(eta-alpha)*k**2)**2/4+0j
        
        
    omega_plus = -0.5j*B+np.sqrt(D)
    omega_minus = -0.5j*B-np.sqrt(D)
    
    return omega_plus,omega_minus,vss


def get_momentum_transCoeffs(kvec,Vec_cqs,Vec_vqs,kappa_axis_qs,Vec_FA_qs,Vec_freqs,Vec_tau_qs,Vec_tauR_qs,Vec_tauN_qs,Nratio_qs,phonon,rots_qpoints,axis=0):
    # calculate second sound velocity in Angstrom*THz = 100 m/s
    # kvec in Angstrom
    eps = 1e-50
    qpoints = phonon.get_mesh_dict()['qpoints']
    Ns = phonon.get_unitcell().get_number_of_atoms()*3
    reclat = np.linalg.inv( phonon.get_unitcell().get_cell())*2*np.pi
    
    C = np.sum(Vec_cqs) # EV/Angstrom**3/K the unmodifed heat capacity
    #convert = (EV/Angstrom**3)*(Angstrom**2*THz)
    #kappa = kappa_cvF[axis,axis] / convert # in EV*THz/Angstrom/K
    

    #Kn = np.outer(kvec,Vec_FA_qs[axis]) # (Nkvec, Nqs)
    #SuppFunc = 1/(1+Kn**2) # suppression function
    
    Pi = 0 # momentum flux Pi = Cvq/w*Nratio
    A = 0 # momentum tensor Cq^2 /w^2   
    M = 0 # phonon viscosity
    Gamma = 0# momentum damping rate
    Ceff = 0 # effective heat capcity corrected by the finite size.
    
    if (type(kvec) == float or type(kvec) == int or type(kvec) == np.float64):
        Nkvec = 1
    else:
        Nkvec = len(kvec)
        
    SuppFuncS = np.zeros((Nkvec,len(Vec_vqs[axis])))
    for iq,qfrac in enumerate(qpoints):
        rots_at_q = rots_qpoints[iq]
        multi_q = len(rots_at_q)
        omegas_at_q = Vec_freqs[iq*Ns:(iq+1)*Ns]*2*np.pi+eps
        cs_at_q = Vec_cqs[iq*Ns:(iq+1)*Ns]
        vs_at_q = Vec_vqs[:,iq*Ns:(iq+1)*Ns]
        Nratios_at_q = Nratio_qs[iq*Ns:(iq+1)*Ns] 
        taus_at_q = Vec_tau_qs[iq*Ns:(iq+1)*Ns]
        tauRs_at_q = Vec_tauR_qs[iq*Ns:(iq+1)*Ns] 
        tauRs_at_q[tauRs_at_q==0] = 1e50
        tauNs_at_q = Vec_tauN_qs[iq*Ns:(iq+1)*Ns] 
        tauNs_at_q[tauNs_at_q==0] = 1e50
        FAs_at_q = Vec_FA_qs[:,iq*Ns:(iq+1)*Ns]
        #SuppFuncs_at_q = SuppFunc[:,iq*Ns:(iq+1)*Ns]
        q = np.dot(reclat,qfrac)                
        
        SuppFuncS_at_q = np.zeros((Nkvec,Ns))
        
        
        for rot in rots_at_q:
            r_cart = similarity_transformation(reclat, rot)
            rot_vs = np.dot(r_cart,vs_at_q)
            rot_Fs = np.dot(r_cart,FAs_at_q)
            rot_q = np.dot(r_cart,q)
            
            Kn_rot_q = np.outer(kvec,rot_vs[axis]*taus_at_q)
            SuppFuncS_at_rotq = 1/(1+Kn_rot_q**2) #SuppFunc[:,iq*Ns:(iq+1)*Ns]
            
            SuppFuncS_at_q += SuppFuncS_at_rotq/multi_q
            
            fac_N = Nratios_at_q*(2-Nratios_at_q)

            Pi += np.matmul(SuppFuncS_at_rotq,cs_at_q*rot_q[axis]*rot_Fs[axis]/omegas_at_q/tauNs_at_q)/multi_q
            A += np.matmul(SuppFuncS_at_rotq,fac_N*cs_at_q*rot_q[axis]**2/omegas_at_q**2)/multi_q
            
            M  += np.matmul(SuppFuncS_at_rotq,cs_at_q*rot_q[axis]**2/omegas_at_q**2*rot_vs[axis]*rot_Fs[axis]/tauNs_at_q)/multi_q
            sum_G_rotq =cs_at_q*(rot_q[axis]**2)/(omegas_at_q**2)*Nratios_at_q/tauRs_at_q/multi_q
            
            Gamma += np.matmul(SuppFuncS_at_rotq,sum_G_rotq)
            Ceff += np.matmul(SuppFuncS_at_rotq,cs_at_q)/multi_q
            
        
        SuppFuncS[:,iq*Ns:(iq+1)*Ns] = SuppFuncS_at_q
    
    kappa = np.matmul(SuppFuncS,kappa_axis_qs)
    
    
            
    alpha = kappa/Ceff
    v0 = np.sqrt(Pi**2/A/Ceff)
    gamma = Gamma/A
    #print(Pi,A,C)
    eta = M/A
    
    

        
    return alpha,v0,eta,gamma
            

def get_vel_ss_intrinsic(Vec_cqs,Vec_vqs,Vec_freqs,phonon,rots_qpoints,axis=0):
    # calculate second sound velocity in Angstrom*THz = 100 m/s
    eps = 1e-50
    qpoints = phonon.get_mesh_dict()['qpoints']
    Ns = phonon.get_unitcell().get_number_of_atoms()*3
    reclat = np.linalg.inv( phonon.get_unitcell().get_cell())*2*np.pi
    
    C = np.sum(Vec_cqs)
    
    Cqv_w = 0
    Cq2_w2 = 0

 
    for iq,qfrac in enumerate(qpoints):
        rots_at_q = rots_qpoints[iq]
        multi_q = len(rots_at_q)
        
        omegas_at_q = Vec_freqs[iq*Ns:(iq+1)*Ns]*2*np.pi+eps
        cs_at_q = Vec_cqs[iq*Ns:(iq+1)*Ns]
        vs_at_q = Vec_vqs[:,iq*Ns:(iq+1)*Ns]        
        q = np.dot(reclat,qfrac)                
        
        for rot in rots_at_q:
            r_cart = similarity_transformation(reclat, rot)
            rot_vs = np.dot(r_cart,vs_at_q)            
            
            Cqv_w += np.sum(cs_at_q*q[axis]*rot_vs[axis]/omegas_at_q)/multi_q
            Cq2_w2 += np.sum(cs_at_q*q[axis]**2/omegas_at_q**2)/multi_q
            
    vss = Cqv_w**2/Cq2_w2/C    
    return vss


# ----------------------------------- Use Full Scattering Matrix to compute VHE coeffs ------------------------------------#

def load_ShengBTE_Phonons_FSM(T0,Dir_BTE_HarPhons,Dir_BTE_MFD,cell,Gamma_cut=True):
    # Gamma_cut, remove the 3 bulk translation modes at Gamma point.
    qpoints_full_list = np.loadtxt(Dir_BTE_HarPhons+'BTE.qpoints_full')
    qpoints_full = qpoints_full_list[:,2:]
    
    # wrap into the 1st BZ.
    qpoints_full[qpoints_full>0.5] -=1
    qpoints_full[qpoints_full<-0.5] +=1
    
    omega_qs = np.loadtxt(Dir_BTE_HarPhons+'BTE.omega_full')
    (Nq,Ns) = omega_qs.shape
    freqs = omega_qs/2/np.pi
    
    cqseV = mode_cv(T0,freqs*THzToEv) 
    cqseV[np.isinf(cqseV)] = 0
    cqseV[np.isnan(cqseV)]  = 0
    
    
    Vol = np.abs(np.linalg.det(cell))
    cqseV = cqseV/Nq/Vol
    cqseV = cqseV.reshape((Nq*Ns,))
    
    gv_in = np.loadtxt(Dir_BTE_HarPhons+'BTE.v_full')*10
    gvx_modes = np.reshape(gv_in[:,0],(Ns,Nq)).T.reshape((Nq*Ns,))
    gvy_modes = np.reshape(gv_in[:,1],(Ns,Nq)).T.reshape((Nq*Ns,))
    gvz_modes = np.reshape(gv_in[:,2],(Ns,Nq)).T.reshape((Nq*Ns,))
    
    gv_full = np.array([gvx_modes,gvy_modes,gvz_modes])
    
    Vec_freqs=freqs.reshape((Nq*Ns,))
    
    if Gamma_cut:
        return qpoints_full,cqseV[3:],gv_full[:,3:],Vec_freqs[3:]
    else:
        return qpoints_full,cqseV,gv_full,Vec_freqs

   
def get_CollMat_W(CollMat_Omega,Vec_freqs):
    # This calculates scattering matrix for phonon energy.
    # Invert W for thermal conductivity.
    
    B = np.outer(Vec_freqs,1/Vec_freqs)
    Wmat = CollMat_Omega*B
    return Wmat 

def get_GreensFunc_1D_FSM(XIx,OMEGAH,Vec_freqs,Vec_cqs,Vec_vqs,Wmat,T0,phonon,qpoints_full,x_direct=0,Xdtype='complex64'):
    
    
    ix = x_direct

    mesh_shape = (XIx.shape[0],OMEGAH.shape[1])
    
    Trise = np.zeros(mesh_shape,dtype=Xdtype)

    
    # Vec_wqs = Vec_freqs*2*np.pi
    # reclat = np.linalg.inv( phonon.get_unitcell().get_cell())*2*np.pi
    # Ns = phonon.get_unitcell().get_number_of_atoms()*3
    # Nq = len(qpoints_full)
    Q_qs = Phonon_GenRate(Vec_cqs,Vec_freqs)
    Vec_vx_qs = Vec_vqs[ix]
    
    Nxir,Nw = mesh_shape
    for ixi in range(Nxir):
        for jw in range(Nw): 
            omega = OMEGAH[0,jw] # I used dense meshgrid.
            xix = XIx[ixi,0]
            
            w_plus_Xiv = omega+xix*Vec_vx_qs
            
            Amat = 1j*np.diag(w_plus_Xiv)+Wmat
            
            #invA = np.linalg.inv(Amat)
            
            Xnum_qs = np.linalg.solve(Amat,Q_qs)
            
            Num = np.sum(Xnum_qs)
            
            Xden_qs = np.linalg.solve(Amat,Vec_cqs*w_plus_Xiv)
            Den = 1j*np.sum(Xden_qs)
            
            Trise[ixi,jw] = Num/Den
            
            
    return Trise
    
    
    
    
    #Amat = np.eye(len(Vec_freqs))
    
    
    

# Pi_ij, A_ij, Mu_ijkl, kappa_ij are calculated using FSM.
def get_Pi_ij_FSM(phonon,Nratio_qs,Vec_cqs,Vec_freqs,qpoints_full,Vec_vqs,dim=1):
    # momentum flux
    # dim is the dimension of the heat tranfer directions
    if dim == 3:
        ij_list = [[0,0],[0,1],[0,2],[1,2],[2,2]]
    elif dim == 2:
        ij_list = [[0,0],[0,1],[1,1]]
    else:
        dim = 1 # else set to 1.
        ij_list = [[0,0]] # defualt is to set as 1D
    
    Vec_omega_qs = Vec_freqs*2*np.pi
    reclat = np.linalg.inv(phonon.get_unitcell().get_cell())*2*np.pi
    Nq = len(qpoints_full)
    Ns = phonon.get_unitcell().get_number_of_atoms()*3       
    qpoints_real = np.matmul(reclat,qpoints_full.T) # real coordinates, 3*Nq

    Pi = np.zeros((dim,dim))
    
    for ij in ij_list:
        i,j = ij
        qi = qpoints_real[i]
        qimat = np.broadcast_to(qi,(Ns,)+qi.shape)
        qi_flat = qimat.T.reshape((Nq*Ns,))[3:] 

        qj = qpoints_real[j]
        qjmat = np.broadcast_to(qj,(Ns,)+qj.shape)
        qj_flat = qjmat.T.reshape((Nq*Ns,))[3:]        
        
        vi = Vec_vqs[i]
        vj = Vec_vqs[j]
        Cvj_w = Vec_cqs/Vec_omega_qs*vj
        Pi_ij = np.dot(qi_flat,Cvj_w)
        if j != i:
            Cvi_w = Vec_cqs/Vec_omega_qs*vi
            Pi_ji = np.dot(qj_flat,Cvi_w)
            Pi[j,i] = Pi_ji
        
        Pi[i,j] = Pi_ij
        
    if (dim == 3 or dim ==2):
        return Pi
    else:
        return Pi[0,0]
    
def get_kappa_ij_FSM(Wmat,Vec_cqs,Vec_vqs,dim=1):
    # thermal conductivity by solving linearized BTE
    
    if dim == 3:
        ij_list = [[0,0],[0,1],[0,2],[1,2],[2,2]]
    elif dim == 2:
        ij_list = [[0,0],[0,1],[1,1]]
    else:
        dim = 1 # defualt is to set as 1D
        ij_list = [[0,0]] # defualt is to set as 1D   
    
    #inv_Wmat = np.linalg.inv(Wmat)
    kappa = np.zeros((dim,dim))
    
    for ij in ij_list:
        i,j = ij
        vi = Vec_vqs[i]
        Cvj = Vec_cqs*Vec_vqs[j]
        
        Xj = np.linalg.solve(Wmat,Cvj)
        kappa_ij = np.dot(vi,Xj)
        #kappa_ij = np.dot(vi,np.matmul(inv_Wmat,Cvj))
        kappa[i,j] = kappa_ij 
        
        if i != j:
            kappa[j,i] = kappa_ij
    
        #kappa_ij = np.dot(vi,np.matmul(invW,Cvj))
    if (dim == 3 or dim ==2):
        return kappa
    else:
        return kappa[0,0]
    

def Gamma_A_ij_FSM(phonon,CollMatR_Omega,Nratio_qs,qpoints_full,Vec_cqs,Vec_freqs,dim=1):
    #i,j=ij    
    
    if dim == 3:
        ij_list = [[0,0,0,0],[0,1],[0,2],[1,2],[2,2]]
    elif dim == 2:
        ij_list = [[0,0],[0,1],[1,1]]
    else:
        dim = 1 # defualt is to set as 1D
        ij_list = [[0,0]] # defualt is to set as 1D
    
    reclat = np.linalg.inv(phonon.get_unitcell().get_cell())*2*np.pi
    qpoints_real = np.matmul(reclat,qpoints_full.T) # real coordinates, 3*Nq
    
    Nq = len(qpoints_full)
    Ns = phonon.get_unitcell().get_number_of_atoms()*3
    Vec_omega_qs = Vec_freqs*2*np.pi
    
    A = np.zeros((dim,dim))
    Gamma = np.zeros((dim,dim))
    
    for ij in ij_list:
        i,j = ij
        qi = qpoints_real[i]
        qj = qpoints_real[j]
    
    
        qimat = np.broadcast_to(qi,(Ns,)+qi.shape)
        qi_flat = qimat.T.reshape((Nq*Ns,))[3:]
    
        qjmat = np.broadcast_to(qj,(Ns,)+qj.shape)
        qj_flat = qjmat.T.reshape((Nq*Ns,))[3:]
    
        Cqj_w2 = Vec_cqs/Vec_omega_qs**2*qj_flat
    

        Gamma_ij = np.dot(qi_flat,np.matmul(CollMatR_Omega,Nratio_qs*Cqj_w2))
        
        Gamma[i,j] = Gamma_ij
        Gamma[j,i] = Gamma_ij
        
    
        A_ij  = np.dot(qi_flat,Cqj_w2)
        A[i,j] = A_ij
        A[j,i] = A_ij
        
    if (dim == 3 or dim ==2):
        return Gamma,A
    else:
        return Gamma[0,0],A[0,0]

def Voigt_to_ijkl(Voigt_tuple,dim=3):
    iv,jv = Voigt_tuple
    
    if dim == 3:
        Voigt_map = [[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]]
        ij = Voigt_map[iv]
        kl = Voigt_map[jv]
    elif dim == 2:
        Voigt_map = [[0,0],[1,1],[0,1]]
        ij = Voigt_map[iv]
        kl = Voigt_map[jv]
    else:
        ij = [0,0]
        kl = [0,0]
    # return two tuples in cartesian notation index.
    return ij,kl
    
def get_phon_viscosity_ijkl_FSM(phonon,CollMat_Omega,Vec_cqs,Vec_freqs,Vec_vqs,qpoints_full,dim=1):
    # 3D Voigt notation: 00 -> 0, 11 -> 1, 22 -> 2, 01 ->3, 12->4 02->5
    # 2D Voigt notation: 00 -> 0, 11 -> 1, 01 -> 2 
    # 1D Voigt notation: 00 -> 0
    #invOmega = np.linalg.inv(CollMat_Omega)
    if dim == 3:
        Voigt_indices = [[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],
                         [1,1],[1,2],[1,3],[1,4],[1,5],
                         [2,2],[2,3],[2,4],[2,5],
                         [3,3],[3,4],[3,5],
                         [4,4],[4,5],[5,5]]
        Mu = np.zeros((6,6))
    elif dim == 2:
        Voigt_indices = [[0,0],[0,1],[0,2],[1,1],[1,2],[2,2]]
        Mu = np.zeros((3,3))
    else:
        dim = 1
        Voigt_indices = [[0,0]]
        Mu = np.zeros((1,1))
    
    reclat = np.linalg.inv(phonon.get_unitcell().get_cell())*2*np.pi
    qpoints_real = np.matmul(reclat,qpoints_full.T) # real coordinates, 3*Nq
    
    Nq = len(qpoints_full)
    Ns = phonon.get_unitcell().get_number_of_atoms()*3
    Vec_omega_qs = Vec_freqs*2*np.pi
    
    for ij_voigt in Voigt_indices:
        iv,jv = ij_voigt
        ij,kl = Voigt_to_ijkl(ij_voigt,dim)
        i,j = ij
        k,l = kl
    
        qi = qpoints_real[i]
        qj = qpoints_real[j]    
     
        qimat = np.broadcast_to(qi,(Ns,)+qi.shape)
        qimat_flat = qimat.T.reshape((Nq*Ns,))[3:]  
        qjmat = np.broadcast_to(qj,(Ns,)+qj.shape)
        qjmat_flat = qjmat.T.reshape((Nq*Ns,))[3:]
    
        Cqjvl_w2 = Vec_cqs*Vec_vqs[l]*qjmat_flat/Vec_omega_qs**2
        qivk = qimat_flat*Vec_vqs[k]
        
        X = np.linalg.solve(CollMat_Omega,Cqjvl_w2) # replace the matrix inversion with this 
    
        Mu_ijkl = np.dot(qivk,X)
        Mu[iv,jv] = Mu_ijkl
    
    if (dim==3 or dim==2):
        return Mu
    else:
        return Mu[0,0]    
    
    
    
