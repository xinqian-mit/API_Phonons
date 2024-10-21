import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation
from numba import njit
import API_phonopy as api_ph
from phonopy.units import VaspToTHz, EV, Angstrom, Kb, THz, THzToEv, Hbar
import subprocess
from os import path
from phonopy.phonon.degeneracy import degenerate_sets


def read_ShengBTE_scattRate(SCATTRATE_FILE,phonon):
    # import and reshape scattering_rate files
    # phonopy objects are input for reshaping the matrix
    freqs = phonon.get_mesh_dict()['frequencies']
    scatt_rate_ph = np.loadtxt(SCATTRATE_FILE)[:,-1]
    scatt_rate_ph = np.reshape(scatt_rate_ph,freqs.T.shape).T
    scatt_rate_ph[np.isnan(scatt_rate_ph)] = np.inf
    
    return scatt_rate_ph


def Tau_modepairs_ShengBTE_q(freqs_THz,Scatt_Rate):
    
    """
    ShengBTE linewidths to compute mode-transition time among mode pairs.
    """
    
    gamma = Scatt_Rate/2.0
    
    Ws,Wr = np.meshgrid(freqs_THz*2*np.pi,freqs_THz*2*np.pi)
    Gs,Gr = np.meshgrid(gamma, gamma) # convert to linewidths.
    
    Num = (Gs+Gr)
    Num[np.isnan(Num)] = 0
    Den = (Ws-Wr)**2+(Gs+Gr)**2 
    Den[Den ==0] = np.inf
    Den[np.isnan(Den)] = np.inf
    Tau_sr = Num/Den # ps
    
    Tau_sr[np.isnan(Tau_sr)] = 0
    Tau_sr[np.isinf(Tau_sr)] = 0
    

    return Tau_sr

def calc_QHGK_ShengBTE_at_T(phonon,mesh,scatt_rate_ph,T):

    freqs = phonon.get_mesh_dict()['frequencies']
    weights = phonon.get_mesh_dict()['weights']
    qpoints = phonon.get_mesh_dict()['qpoints'] # irreducible wedge
    
    unit_to_WmK = (Angstrom*THz)**2 /(Angstrom**3)/THz  # need to check this factor. May have an extra 2pi.

    Vol = phonon.get_primitive().get_volume()

    Nq = len(qpoints) # number of irreducible qpoints.
    Ns = 3*phonon.get_primitive().get_number_of_atoms() # number of phonon branches.

    C_mp = np.zeros((Nq,Ns,Ns))# specific heat of mode pairs (mp)

    # symmetrize kijq_modes
    Rot_lists = phonon.get_symmetry().get_symmetry_operations()['rotations']
    Nrots = len(Rot_lists)

    Kappa_Kubo = np.zeros(6) #Kubo thermal conductivity (xx,yy,zz,xy,yz,zx)
    Kappa_Ph =np.zeros(6) # quasiparticle picture of phonon gas.
    
    # Here I explictly show how to compute thermal conductivity using Kubo Formula.
    
    Kxx_mp = []
    Kyy_mp = []
    Kzz_mp = []

    for iq,q in enumerate(qpoints):
        weight_q = weights[iq]
        q = qpoints[iq]
        C_mp_q = calc_Cv_modepairs_q(np.abs(freqs[iq]),T)/Vol/np.prod(mesh) # specific heat mode pairs at a q point. 
        C_mp[iq] = C_mp_q*weight_q

        gvm_q = get_velmat_modepairs_q(phonon,q) # group velocity operator at q.

        gvm_by_gvm_q = get_velmat_by_velmat_q(gvm_q,phonon,q)

        Tau_mp_q = Tau_modepairs_ShengBTE_q(freqs[iq],scatt_rate_ph[iq])

        Kxxq_modes = np.real(C_mp_q*gvm_by_gvm_q[0]*Tau_mp_q)*weight_q*unit_to_WmK
        Kyyq_modes = np.real(C_mp_q*gvm_by_gvm_q[1]*Tau_mp_q)*weight_q*unit_to_WmK
        Kzzq_modes = np.real(C_mp_q*gvm_by_gvm_q[2]*Tau_mp_q)*weight_q*unit_to_WmK
        Kxyq_modes = np.real(C_mp_q*gvm_by_gvm_q[3]*Tau_mp_q)*weight_q*unit_to_WmK
        Kyzq_modes = np.real(C_mp_q*gvm_by_gvm_q[4]*Tau_mp_q)*weight_q*unit_to_WmK
        Kxzq_modes = np.real(C_mp_q*gvm_by_gvm_q[5]*Tau_mp_q)*weight_q*unit_to_WmK


        # Kxxq_modes_sym = np.zeros_like(Kxxq_modes)
        # Kyyq_modes_sym = np.zeros_like(Kyyq_modes)
        # Kzzq_modes_sym = np.zeros_like(Kzzq_modes)
        # Kxyq_modes_sym = np.zeros_like(Kxyq_modes)
        # Kyzq_modes_sym = np.zeros_like(Kyzq_modes)
        # Kxzq_modes_sym = np.zeros_like(Kxzq_modes)
        # for rot in Rot_lists:
        #     invrot = np.linalg.inv(rot)

        #     RK_xx = rot[0,0]*Kxxq_modes + rot[0,1]*Kxyq_modes + rot[0,2]*Kxzq_modes # xx*xx xy*yx xz*zx
        #     RK_xy = rot[0,0]*Kxyq_modes + rot[0,1]*Kyyq_modes + rot[0,2]*Kyzq_modes # xx*xy xy*yy xz*zy
        #     RK_xz = rot[0,0]*Kxzq_modes + rot[0,1]*Kyzq_modes + rot[0,2]*Kzzq_modes # xx*xz xy*yz xz*zz
        #     RK_yx = rot[1,0]*Kxxq_modes + rot[1,1]*Kxyq_modes + rot[1,2]*Kxzq_modes # yx*xx yy*yx yz*zx
        #     RK_yy = rot[1,0]*Kxyq_modes + rot[1,1]*Kyyq_modes + rot[1,2]*Kyzq_modes # yx*xy yy*yy yz*zy
        #     RK_yz = rot[1,0]*Kxzq_modes + rot[1,1]*Kyzq_modes + rot[1,2]*Kzzq_modes # yx*xz yy*yz yz*zz        
        #     RK_zx = rot[2,0]*Kxxq_modes + rot[2,1]*Kxyq_modes + rot[2,2]*Kxzq_modes # yx*xx yy*yx yz*zx
        #     RK_zy = rot[2,0]*Kxyq_modes + rot[2,1]*Kyyq_modes + rot[2,2]*Kyzq_modes # yx*xy yy*yy yz*zy
        #     RK_zz = rot[2,0]*Kxzq_modes + rot[2,1]*Kyzq_modes + rot[2,2]*Kzzq_modes # yx*xz yy*yz yz*zz     

        #     R_K_invR_xx = RK_xx*invrot[0,0] + RK_xy*invrot[1,0] + RK_xz*invrot[2,0]
        #     R_K_invR_xy = RK_xx*invrot[0,1] + RK_xy*invrot[1,1] + RK_xz*invrot[2,1]
        #     R_K_invR_xz = RK_xx*invrot[0,2] + RK_xy*invrot[1,2] + RK_xz*invrot[2,2]
        #     R_K_invR_yx = RK_yx*invrot[0,0] + RK_yy*invrot[1,0] + RK_yz*invrot[2,0]
        #     R_K_invR_yy = RK_yx*invrot[0,1] + RK_yy*invrot[1,1] + RK_yz*invrot[2,1]
        #     R_K_invR_yz = RK_yx*invrot[0,2] + RK_yy*invrot[1,2] + RK_yz*invrot[2,2]
        #     R_K_invR_zx = RK_zx*invrot[0,0] + RK_zy*invrot[1,0] + RK_zz*invrot[2,0]
        #     R_K_invR_zy = RK_zx*invrot[0,1] + RK_zy*invrot[1,1] + RK_zz*invrot[2,1]
        #     R_K_invR_zz = RK_zx*invrot[0,2] + RK_zy*invrot[1,2] + RK_zz*invrot[2,2]

        #     Kxxq_modes_sym += R_K_invR_xx
        #     Kyyq_modes_sym += R_K_invR_yy
        #     Kzzq_modes_sym += R_K_invR_zz
        #     Kxyq_modes_sym += (R_K_invR_xy + R_K_invR_yx)/2
        #     Kyzq_modes_sym += (R_K_invR_yz + R_K_invR_zy)/2
        #     Kxzq_modes_sym += (R_K_invR_xz + R_K_invR_zx)/2


        # Kxxq_modes = Kxxq_modes_sym/Nrots
        # Kyyq_modes = Kyyq_modes_sym/Nrots
        # Kzzq_modes = Kzzq_modes_sym/Nrots
        # Kxyq_modes = Kxyq_modes_sym/Nrots
        # Kyzq_modes = Kyzq_modes_sym/Nrots
        # Kxzq_modes = Kxzq_modes_sym/Nrots
        
        Kxx_mp.append(Kxxq_modes)
        Kyy_mp.append(Kyyq_modes)
        Kzz_mp.append(Kzzq_modes)

        Kappa_Kubo[0] += np.sum(Kxxq_modes)
        Kappa_Kubo[1] += np.sum(Kyyq_modes)
        Kappa_Kubo[2] += np.sum(Kzzq_modes)
        Kappa_Kubo[3] += np.sum(Kxyq_modes)
        Kappa_Kubo[4] += np.sum(Kyzq_modes)
        Kappa_Kubo[5] += np.sum(Kxzq_modes)

        Kappa_Ph[0] += np.trace(Kxxq_modes) # diagonal part corresponds to quasi-particles
        Kappa_Ph[1] += np.trace(Kyyq_modes)
        Kappa_Ph[2] += np.trace(Kzzq_modes)
        Kappa_Ph[3] += np.trace(Kxyq_modes)
        Kappa_Ph[4] += np.trace(Kyzq_modes)
        Kappa_Ph[5] += np.trace(Kxzq_modes)
        
    return Kappa_Kubo,Kappa_Ph,np.array(Kxx_mp),np.array(Kyy_mp),np.array(Kzz_mp),freqs




def calc_QHGK_phono3py(phonon,mesh,Temperatures,load=True,nac=False,lbte=False): # multiple temperatures
    Nrepeat = phonon.get_supercell_matrix().diagonal()
    
    
    if type(Temperatures) == float or type(Temperatures) == int:
        Temperatures = np.array([Temperatures]) #convert to list


    if type(Temperatures) == list or type(Temperatures) == tuple:             
        Temperatures = np.array(Temperatures)
 
    if nac:
        phono3py_cmd = 'phono3py --dim="{} {} {}" --fc2 --fc3 --nac --br --mesh="'\
                   '{} {} {}" --ts="{}" > ph3.out'.format(Nrepeat[0],Nrepeat[1],Nrepeat[2], 
                                                 mesh[0],mesh[1],mesh[2], ' '.join(str(T) for T in Temperatures))  
        if lbte:
            phono3py_cmd = 'phono3py --dim="{} {} {}" --fc2 --fc3 --nac --br --lbte --mesh="'\
                       '{} {} {}" --ts="{}" > ph3.out'.format(Nrepeat[0],Nrepeat[1],Nrepeat[2], 
                                                     mesh[0],mesh[1],mesh[2],' '.join(str(T) for T in Temperatures))  
    else:
        phono3py_cmd = 'phono3py --dim="{} {} {}" --fc2 --fc3 --br --mesh="'\
                   '{} {} {}" --ts="{}" > ph3.out'.format(Nrepeat[0],Nrepeat[1],Nrepeat[2], 
                                                 mesh[0],mesh[1],mesh[2], ' '.join(str(T) for T in Temperatures))  
                                                 
        if lbte:
            phono3py_cmd = 'phono3py --dim="{} {} {}" --fc2 --fc3 --br --lbte --mesh="'\
                       '{} {} {}" --ts="{}" > ph3.out'.format(Nrepeat[0],Nrepeat[1],Nrepeat[2], 
                                                     mesh[0],mesh[1],mesh[2], ' '.join(str(T) for T in Temperatures))    
                                                                                                 

    if load == False:
        subprocess.call(phono3py_cmd, shell=True)
        qpoints,weights,freqs,gamma,kappaT = api_ph.read_phono3py_hdf5(mesh)
        
    if load == True:
        if mesh[0]==0 & mesh[1]==0 & mesh[2]==0:
            filename = 'kappa-m{}{}{}-g0.hdf5'.format(mesh[0],mesh[1],mesh[2])
        else:
            filename = 'kappa-m{}{}{}.hdf5'.format(mesh[0],mesh[1],mesh[2])
        
        if path.exists(filename):
            qpoints,weights,freqs,gamma,kappaT = api_ph.read_phono3py_hdf5(mesh)
        
        else:
            subprocess.call(phono3py_cmd, shell=True)
            qpoints,weights,freqs,gamma,kappaT = api_ph.read_phono3py_hdf5(mesh)
            
    phonon.set_mesh(mesh)
    unit_to_WmK = (Angstrom*THz)**2 /(Angstrom**3)


    #CV = 0

    #K = 0


    Vol = phonon.get_primitive().get_volume() # get volume.


    #filename = 'kappa-m{}{}{}.hdf5'.format(mesh[0],mesh[1],mesh[2])
    #ph3_data = h5py.File(filename,'r')

    #CV_qmodes = ph3_data['heat_capacity'][:][0]/Vol/np.prod(mesh)*EV
    
    kxx = np.zeros(Temperatures.shape)
    kyy = np.zeros(Temperatures.shape)
    kzz = np.zeros(Temperatures.shape)
    kxy = np.zeros(Temperatures.shape)
    kyz = np.zeros(Temperatures.shape)
    kxz = np.zeros(Temperatures.shape)
    
    kxx_ph = np.zeros(Temperatures.shape)
    kyy_ph = np.zeros(Temperatures.shape)
    kzz_ph = np.zeros(Temperatures.shape)
    kxy_ph = np.zeros(Temperatures.shape)
    kyz_ph = np.zeros(Temperatures.shape)
    kxz_ph = np.zeros(Temperatures.shape)
    
    gvm_by_gvm =[]
    
    # compute gvm_by_gvm which is independent of T
    
    #print(len(qpoints))

    for iq,q in enumerate(qpoints):
        
        # temperature independent properties.
        gvm = get_velmat_modepairs_q(phonon,q) 
        gvm_by_gvm_q = get_velmat_by_velmat_q(gvm,phonon,q)
        gvm_by_gvm.append(gvm_by_gvm_q)
        
        
    
    Kxx_mp = []
    Kyy_mp = []
    Kzz_mp = []
    
    kappa = np.zeros((len(Temperatures),3,3))
    kappa_ph = np.zeros((len(Temperatures),3,3))
    
        
    for iT,T in enumerate(Temperatures):
        
        KxxT_mp = []
        KyyT_mp = []
        KzzT_mp = []
        
        for iq,q in enumerate(qpoints):
            weight_q = weights[iq]
            freqs_q = freqs[iq]
            gamma_q = gamma[iT][iq] # in THz, need to convert to Trad/s.
            
            gamma_q[gamma_q ==0 ] = np.inf
            C_mp_q = calc_Cv_modepairs_q(freqs_q,T)/Vol/np.prod(mesh) # pairwise specific heat.
            
            Tau_mp = Tau_modepairs_ph3_q(freqs_q,gamma_q) # in seconds
            Kxxq_modes = np.real(C_mp_q*gvm_by_gvm[iq][0]*Tau_mp)*weight_q*unit_to_WmK
            Kyyq_modes = np.real(C_mp_q*gvm_by_gvm[iq][1]*Tau_mp)*weight_q*unit_to_WmK
            Kzzq_modes = np.real(C_mp_q*gvm_by_gvm[iq][2]*Tau_mp)*weight_q*unit_to_WmK
            Kxyq_modes = np.real(C_mp_q*gvm_by_gvm[iq][3]*Tau_mp)*weight_q*unit_to_WmK
            Kyzq_modes = np.real(C_mp_q*gvm_by_gvm[iq][4]*Tau_mp)*weight_q*unit_to_WmK
            Kxzq_modes = np.real(C_mp_q*gvm_by_gvm[iq][5]*Tau_mp)*weight_q*unit_to_WmK
            
            Kxxq_modes_sym = np.zeros_like(Kxxq_modes)
            Kyyq_modes_sym = np.zeros_like(Kyyq_modes)
            Kzzq_modes_sym = np.zeros_like(Kzzq_modes)
            Kxyq_modes_sym = np.zeros_like(Kxyq_modes)
            Kyzq_modes_sym = np.zeros_like(Kyzq_modes)
            Kxzq_modes_sym = np.zeros_like(Kxzq_modes)
            
            #print(Kxxq_modes.shape)
            
            
            # symmetrize kijq_modes
            Rot_lists = phonon.get_symmetry().get_symmetry_operations()['rotations']
            Nrots = len(Rot_lists)
            
            # Ksym = R*K*inv(R) # symmetrize the thermal conductivity tensor        
            for rot in Rot_lists:
                invrot = np.linalg.inv(rot)

                RK_xx = rot[0,0]*Kxxq_modes + rot[0,1]*Kxyq_modes + rot[0,2]*Kxzq_modes # xx*xx xy*yx xz*zx
                RK_xy = rot[0,0]*Kxyq_modes + rot[0,1]*Kyyq_modes + rot[0,2]*Kyzq_modes # xx*xy xy*yy xz*zy
                RK_xz = rot[0,0]*Kxzq_modes + rot[0,1]*Kyzq_modes + rot[0,2]*Kzzq_modes # xx*xz xy*yz xz*zz
                RK_yx = rot[1,0]*Kxxq_modes + rot[1,1]*Kxyq_modes + rot[1,2]*Kxzq_modes # yx*xx yy*yx yz*zx
                RK_yy = rot[1,0]*Kxyq_modes + rot[1,1]*Kyyq_modes + rot[1,2]*Kyzq_modes # yx*xy yy*yy yz*zy
                RK_yz = rot[1,0]*Kxzq_modes + rot[1,1]*Kyzq_modes + rot[1,2]*Kzzq_modes # yx*xz yy*yz yz*zz        
                RK_zx = rot[2,0]*Kxxq_modes + rot[2,1]*Kxyq_modes + rot[2,2]*Kxzq_modes # yx*xx yy*yx yz*zx
                RK_zy = rot[2,0]*Kxyq_modes + rot[2,1]*Kyyq_modes + rot[2,2]*Kyzq_modes # yx*xy yy*yy yz*zy
                RK_zz = rot[2,0]*Kxzq_modes + rot[2,1]*Kyzq_modes + rot[2,2]*Kzzq_modes # yx*xz yy*yz yz*zz     
                
                R_K_invR_xx = RK_xx*invrot[0,0] + RK_xy*invrot[1,0] + RK_xz*invrot[2,0]
                R_K_invR_xy = RK_xx*invrot[0,1] + RK_xy*invrot[1,1] + RK_xz*invrot[2,1]
                R_K_invR_xz = RK_xx*invrot[0,2] + RK_xy*invrot[1,2] + RK_xz*invrot[2,2]
                R_K_invR_yx = RK_yx*invrot[0,0] + RK_yy*invrot[1,0] + RK_yz*invrot[2,0]
                R_K_invR_yy = RK_yx*invrot[0,1] + RK_yy*invrot[1,1] + RK_yz*invrot[2,1]
                R_K_invR_yz = RK_yx*invrot[0,2] + RK_yy*invrot[1,2] + RK_yz*invrot[2,2]
                R_K_invR_zx = RK_zx*invrot[0,0] + RK_zy*invrot[1,0] + RK_zz*invrot[2,0]
                R_K_invR_zy = RK_zx*invrot[0,1] + RK_zy*invrot[1,1] + RK_zz*invrot[2,1]
                R_K_invR_zz = RK_zx*invrot[0,2] + RK_zy*invrot[1,2] + RK_zz*invrot[2,2]
                
                Kxxq_modes_sym += R_K_invR_xx
                Kyyq_modes_sym += R_K_invR_yy
                Kzzq_modes_sym += R_K_invR_zz
                Kxyq_modes_sym += (R_K_invR_xy + R_K_invR_yx)/2
                Kyzq_modes_sym += (R_K_invR_yz + R_K_invR_zy)/2
                Kxzq_modes_sym += (R_K_invR_xz + R_K_invR_zx)/2
                
                
            Kxxq_modes_sym = Kxxq_modes_sym/Nrots
            Kyyq_modes_sym = Kyyq_modes_sym/Nrots
            Kzzq_modes_sym = Kzzq_modes_sym/Nrots
            Kxyq_modes_sym = Kxyq_modes_sym/Nrots
            Kyzq_modes_sym = Kyzq_modes_sym/Nrots
            Kxzq_modes_sym = Kxzq_modes_sym/Nrots
            
            #print(Kxxq_modes_sym.shape)       
                     
        

            KxxT_mp.append(Kxxq_modes_sym)
            KyyT_mp.append(Kyyq_modes_sym)
            KzzT_mp.append(Kzzq_modes_sym)
    
            kxx[iT] += np.sum(Kxxq_modes_sym)
            kyy[iT] += np.sum(Kyyq_modes_sym)
            kzz[iT] += np.sum(Kzzq_modes_sym)
            kxy[iT] += np.sum(Kxyq_modes_sym)
            kyz[iT] += np.sum(Kyzq_modes_sym)
            kxz[iT] += np.sum(Kxzq_modes_sym)


            kxx_ph[iT] += np.trace(Kxxq_modes_sym)
            kyy_ph[iT] += np.trace(Kyyq_modes_sym)
            kzz_ph[iT] += np.trace(Kzzq_modes_sym)
            kxy_ph[iT] += np.trace(Kxyq_modes_sym)
            kyz_ph[iT] += np.trace(Kyzq_modes_sym)
            kxz_ph[iT] += np.trace(Kxzq_modes_sym)
            
        Kxx_mp.append(np.array(KxxT_mp))
        Kyy_mp.append(np.array(KyyT_mp))
        Kzz_mp.append(np.array(KzzT_mp))

   

        kappa[iT,0,0] = kxx[iT]
        kappa[iT,1,1] = kyy[iT]
        kappa[iT,2,2] = kzz[iT]
        kappa[iT,0,1] = kxy[iT]
        kappa[iT,1,0] = kxy[iT]
        kappa[iT,1,2] = kyz[iT]
        kappa[iT,2,1] = kyz[iT]
        kappa[iT,0,2] = kxz[iT]
        kappa[iT,2,0] = kxz[iT]

    
        kappa_ph[iT,0,0] = kxx_ph[iT]
        kappa_ph[iT,1,1] = kyy_ph[iT]
        kappa_ph[iT,2,2] = kzz_ph[iT]
        kappa_ph[iT,0,1] = kxy_ph[iT]
        kappa_ph[iT,1,0] = kxy_ph[iT]
        kappa_ph[iT,1,2] = kyz_ph[iT]
        kappa_ph[iT,2,1] = kyz_ph[iT]
        kappa_ph[iT,0,2] = kxz_ph[iT]
        kappa_ph[iT,2,0] = kxz_ph[iT]
    
    # For one temperature, Kij_mp[q,s1,s2] is tha pairwise conductivity at q point, between s1 and s2
    # For multiple temperautres Kij_mp[T,q,s1,s2]
    
    
    if len(Temperatures)==1:
        return kappa,kappa_ph,Kxx_mp[0],Kyy_mp[0],Kzz_mp[0],freqs,weights
    
    if len(Temperatures)>1:
        return kappa,kappa_ph,np.array(Kxx_mp),np.array(Kyy_mp),np.array(Kzz_mp),freqs


def get_dq_dynmat_q(phonon,q,dq=1e-5):
    phonon._set_group_velocity()
    _gv = phonon._group_velocity
    ddm = _gv._get_dD(q)
    
    #ddm = [ddmx,ddmy,ddmz]
    return ddm[1:]




def get_velmat_by_velmat_q(gvm,phonon,q,symetrize=False):
    """
    output vnm x vnm. the first dimension is cartisian indices, in the order of 
    xx,yy,zz,xy,yz,xz.

    """
    rots =  phonon.symmetry.reciprocal_operations
    gvm_by_gvm = np.zeros((6,)+gvm.shape[1:],dtype=gvm.dtype)
    
    reclat = np.linalg.inv(phonon.get_primitive().cell)
    multi = 0
    
    rots_sitesym = []
    
    # compute rotational multiplicity.
    
    for rot in rots:
        q_rot = np.dot(rot,q)
        diff = q - q_rot
        diff -= np.rint(diff)
        dist = np.linalg.norm(np.dot(reclat, diff))
        if dist < phonon.symmetry.tolerance:
            multi += 1
            rots_sitesym.append(rot)

    
    for idir,(ii,jj) in enumerate([[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]]):
        for rot in rots_sitesym:
            r_cart = similarity_transformation(reclat, rot)
            r_gvm = np.einsum("ij,jkl->ikl",r_cart,gvm)
            gvm_by_gvm[idir] += np.conj(r_gvm[ii])*r_gvm[jj]
            
        gvm_by_gvm[idir] /= multi # symmetrize gvm_by_gvm
        
    return gvm_by_gvm
        

def get_velmat_modepairs_q(phonon, q, factor=VaspToTHz,cutoff_frequency=1e-4): # suitable for crystalline system.
    
    if np.linalg.norm(q) < cutoff_frequency: # at Gamma point.
        freqs,eigvecs = phonon.get_frequencies_with_eigenvectors(q)
        if (freqs<cutoff_frequency).any():
            print('largest imaginary frequencies:',np.min(freqs[freqs<0])) 
            freqs = np.abs(freqs)
        ddm = get_dq_dynmat_q(phonon,q) # three components.
    
        sqrt_fnfm = np.sqrt(freqs.T*freqs)
    
        temp_vx = np.dot(ddm[0],eigvecs)*factor**2
        vx_modepairs = np.dot(eigvecs.conjugate().T,temp_vx)/sqrt_fnfm/2/(2*np.pi) # ATHz

    
        temp_vy = np.dot(ddm[1],eigvecs)*factor**2
        vy_modepairs = np.dot(eigvecs.conjugate().T,temp_vy)/sqrt_fnfm/2/(2*np.pi) # ATHz

    
        temp_vz = np.dot(ddm[2],eigvecs)*factor**2
        vz_modepairs = np.dot(eigvecs.conjugate().T,temp_vz)/sqrt_fnfm/2/(2*np.pi) # ATHz
        
        gvm = np.array([vx_modepairs,vy_modepairs,vz_modepairs])
        
        
        return np.real(gvm)
    
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
        
        return gvm

        # if _gv._perturbation is None:
        #     if _gv._symmetry is None:
        #         return gvm
        #     else:
        #         if np.linalg.norm(q) == 0: # if at Gamma point, don't symmetrize
        #             return gvm
        #         else:
        #             return symmetrize_group_velocity_matrix(gvm, phonon, q)
        # else:
        #     return gvm

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
    #gvm_sym = (gvm_sym + gvm_sym.transpose(0, 2, 1).conj()) / 2

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
    
    freqs = np.abs(freqs_THz)*THzToEv
    
    if T==0:
        n_modes = 0
        Cs = np.zeros(freqs_THz.shape)
    else:
        
        x = freqs / Kb / T
        expVal = np.exp(x)
        n_modes = 1/(expVal-1.0)
        Cs = Kb * x ** 2 * expVal / (expVal - 1.0) ** 2
        
    Nmodes = len(freqs) #number of modes
    
    Ws,Wr = np.meshgrid(freqs+1e-10,freqs) # small offset
    Ns,Nr = np.meshgrid(n_modes,n_modes)
    
    Csr = Ws*Wr*(Ns-Nr)/(Wr-Ws)/T 
    Csr = Csr - np.diag(Csr.diagonal())
    Csr = Csr + np.diag(Cs) #eV/K
    Csr = Csr*EV # J/K
    
    return Csr
     


def Tau_modepairs_ph3_q(freqs_THz,gamma):
    
    """
    phono3py linewidths to compute mode-transition time among mode pairs.
    """
    
    Ws,Wr = np.meshgrid(freqs_THz*THzToEv,freqs_THz*THzToEv)
    Gs,Gr = np.meshgrid(gamma*THzToEv, gamma*THzToEv) # convert to angular frequency linewidths.
    
    Num =Gs+Gr
    Num[np.isnan(Num)] = 0
    Den = (Ws-Wr)**2+(Gs+Gr)**2
    Den[Den ==0] = np.inf
    Den[np.isnan(Den)] = np.inf
    Tau_sr = Num/Den # ps
    
    
    
    gamma[gamma == 0] =np.inf
    
    tau_s = 1/(gamma*2*THzToEv)
    
    Tau_sr_ndiag = Tau_sr - np.diag(Tau_sr.diagonal())
    Tau_sr = (np.diag(tau_s) + Tau_sr_ndiag)
    
    Tau_sr[np.isnan(Tau_sr)] = 0 # in eV^-1
    
    Tau_sr = Tau_sr*Hbar # Hbar in eVs
    return Tau_sr
