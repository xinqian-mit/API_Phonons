import numpy as np
import sys
from scipy.integrate import simpson #,trapezoid
from scipy.interpolate import PchipInterpolator,CubicSpline	
from scipy.special import jn # Bessel function of the first kind
# --------------------------------------- Fourier and Inverse Fourier Transforms --------------------------------------------------------------#

def Interp_FDsignal(omega,Fw,Ws,Wmax,method='linear'):

    if Wmax > np.max(np.abs(omega)):
        Wmax =  np.max(np.abs(omega))
    
    omega_intp = np.arange(0,Wmax,Ws)
    
    if method == 'pci' or 'PCI':
        
        pci = PchipInterpolator(omega,Fw)
        Fwp_intp = pci(omega_intp)
        
    elif method == 'CubicSpline' :
        cs = CubicSpline(omega,Fw)
        Fwp_intp = cs(omega_intp)
        
    else:    
        Fwp_intp = np.interp(omega_intp,omega,Fw)
    
    return omega_intp,Fwp_intp

def FourierTrans_Even(t,y,Ts):
    # input t>0, do even extrapolation, and then perform Fourier transform
    Tcut = np.max(t)
    tsamp = np.arange(0,Tcut,Ts)
    Nt = len(tsamp) 
    Ws =  2*np.pi/(Nt*Ts) # sampling freq
    
    pci = PchipInterpolator(t,y)
    ysamp = pci(tsamp) 
    
    #ysamp = np.interp(tsamp,t,y) 
       
    # if t not equally spaced, do linear interpolation
    
    omega = np.arange(0,int(Nt/2))*Ws
    phase = np.exp(1j*omega*Tcut)
    
    RawFT_y = np.fft.fft(ysamp)[0:int(Nt/2)]*Ts
    
    FT_y = RawFT_y*phase
    
    return omega,FT_y+FT_y.conj()
    

def InvFourTrans_Even(omegap,Fwp,Ws):
    # do even extrapolation. Ws is the sampling freq to for ifft
    # input omegap>0
    # if Ws corresponds to a denser frequency ticks, then interpolation will be performed. 
    
    Wcut = np.max(omegap) 
    Mp = len(omegap)
    
    omegam = -np.flip(omegap)[:Mp-1]
    Fwm = np.flip(Fwp)[:Mp-1] # even extrapolate to negative freqs
    
    omega_even = np.concatenate((omegam,omegap))
    Fw_even = np.concatenate((Fwm,Fwp))
    
    wsamp = np.arange(-Wcut,Wcut,Ws)
    pci = PchipInterpolator(omega_even,Fw_even)
    Fw_samp = pci(wsamp)
    #Fw_samp = np.interp(wsamp,omega_even,Fw_even)
    
    N = len(wsamp)
    M = int((N+1)/2)


    Ts = np.pi/(M*Ws)
    t = np.arange(0,M)*Ts
    
    yext_ifft = np.fft.ifft(Fw_samp)/Ts*np.exp(-1j*wsamp*(M-1)*Ts) #/1j
    
    y_ifft = yext_ifft[0:M]
    
    #y_ifft = y_ifft+y_ifft.conj()
    #y_ifft /= 2.0
    return t,np.real(y_ifft)


def InvFourTrans_Hermit(omegap,Fwp,Ws):
    # do even extrapolation. Ws is the sampling freq to for ifft
    # input omegap>0
    # if Ws corresponds to a denser frequency ticks, then interpolation will be performed. 
    
    Wcut = np.max(omegap) 
    Mp = len(omegap)
    
    omegam = -np.flip(omegap)[:Mp-1]
    Re_Fwm = np.flip(np.real(Fwp))[:Mp-1] # even extrapolate to negative freqs
    Im_Fwm = -np.flip(np.imag(Fwp))[:Mp-1] # even extrapolate to negative freqs
    Fw_m = Re_Fwm + 1.0j*Im_Fwm
    
    omega_even = np.concatenate((omegam,omegap))
    Fw_H = np.concatenate((Fw_m,Fwp))
    
    wsamp = np.arange(-Wcut,Wcut,Ws)
    pci = PchipInterpolator(omega_even,Fw_H)
    Fw_samp = pci(wsamp)
    #Fw_samp = np.interp(wsamp,omega_even,Fw_even)
    
    N = len(wsamp)
    M = int((N+1)/2)


    Ts = np.pi/(M*Ws)
    t = np.arange(0,M)*Ts
    
    yext_ifft = np.fft.ifft(Fw_samp)/Ts*np.exp(-1j*wsamp*(M-1)*Ts)
    
    y_ifft = yext_ifft[0:M]
    
    #y_ifft = y_ifft+y_ifft.conj()
    #y_ifft /= 2.0
    return t,np.real(y_ifft)

# --------------------------------------- Compute pump-probe signals from Green's functions ---------------------------------------------------#

# For simplicity, we are considering a bulk thermoreflectance measurement.

# Here XIr is the in-plane hankel transform variable, XIz is the through-plane Fourier transform variable. OmegaH is the heating frequency

#

def Gauss_2DFT(XIx,XIy,rxy):
    
    if type(rxy) == float or type(rxy) ==int or type(rxy)==np.float64:
        profile = np.exp(-(XIx*rxy)**2/8)*np.exp(-(XIy*rxy)**2/8)
    else:
        profile = np.exp(-(XIx*rxy[0])**2/8)*np.exp(-(XIy*rxy[1])**2/8)  
    return profile
    

def Gauss_Offset_2DFT(XIx,XIy,rxy,offset_xy):
    x0 = offset_xy[0]
    y0 = offset_xy[1]    
    offsetphase = np.exp(-1j*(XIx*x0+XIy*y0))   
    profile = Gauss_2DFT(XIx,XIy,rxy)*offsetphase
    return profile

 
def Gauss_Hankel(XIr,rp):
    profile = np.exp(-(XIr*rp)**2/8)   
    return profile
    
def get_PolyCoeffs(Nmax=40):

# Polynomial defined by Joseph Feser for offset gaussian beam.
# Rev. Sci. Instrum. 83, 104901 (2012)
 
    Kmax = Nmax*2+2;

    p = np.zeros((Nmax+1,Kmax))
    p[0,-1] = np.pi 

    L = np.array([np.pi**2,0,-1,0])
    M = np.array([-1,0,1/(4*np.pi**2)])
    N = np.array([1/(4*np.pi**2),0])
    G = np.array([-1,0])


    for n in range(Nmax):

        ind = n+1
        pp = p[ind-1,:]
        dpp = np.polyder(pp) # derivative 
        d2pp = np.polyder(dpp) # second derivative
        term1 = np.convolve(L,pp)
        term2 = np.convolve(M,dpp)
        term3 = np.convolve(N,d2pp)
        NUM = np.polyadd(np.polyadd(term1,term2),term3)

        DEN = G
        pnew = np.polydiv(NUM,DEN)[0]

        p[ind,:] = pnew[-Kmax:]
        
    return p
    
def Ring_Hankel(XIr,R0,w):
    
    return np.exp(-(XIr*w)**2/8)*jn(0,XIr*R0)


def Ring_Fourier(XIx,XIy,R0,w):
    
    XIr = np.sqrt(XIx**2+XIy**2)
    
    return np.exp(-(XIr*w)**2/8)*jn(0,XIr*R0)


def RectRing_Hankel(XIr,R0,w):
    
    #XIr += eps
    Num = (R0+w)*XIr*jn(1,(R0+w)*XIr)-(R0-w)*XIr*jn(1,(R0-w)*XIr)
    Den = XIr**2
    
    RecRing_Hankel = Num/Den
    RecRing_Hankel[np.isnan(RecRing_Hankel)] = 0.5*((R0+w)**2-(R0-w)**2)
    
    return RecRing_Hankel


def RectRing_Fourier(XIx,XIy,R0,w):
    
    #XIr += eps
    XIr = np.sqrt(XIx**2+XIy**2)
    Num = (R0+w)*XIr*jn(1,(R0+w)*XIr)-(R0-w)*XIr*jn(1,(R0-w)*XIr)
    Den = XIx**2+XIy**2
    
    Ring_Hankel = Num/Den
    Ring_Hankel[np.isnan(Ring_Hankel)] = 0.5*((R0+w)**2-(R0-w)**2)
    
    return Ring_Hankel



def SensBeam_Hankel(XIr,rs,x0,Nmax=40):
    
    # H(f) = int f(r)J0(xr)rdr. It will have a 2pi factor compared with the 
    # convention H(f) = int f(r)J0(2pixr)2pi*rdr. 
    # x0 is the offset distance from the pump beam.
    
    kvec = XIr/2/np.pi

    kbar = kvec*rs/np.sqrt(2)
    xbar = np.sqrt(2)*x0/rs

    prefactor = 1/np.pi*np.exp(-(xbar**2+np.pi**2*kbar**2))
    x2 = xbar**2

    p = get_PolyCoeffs(Nmax)

    #print(p)

    #Nk = len(kbar)

    #sumterm = np.zeros_like(XIr)
    sigma = np.zeros_like(XIr)




    for n in range(Nmax+1):
        PP = np.polyval(p[n],kbar)
        nfact = np.math.factorial(n)
        
        nfact = np.min([nfact,sys.float_info.max])
        
        summand = x2**n/nfact**2*PP
        #print(summand)
        sigma += summand

    S = (prefactor*sigma)*np.ones_like(XIr)
    
    return S
    
    
# compute temperature rise in the Hankel-Fourier transformation domain. 
def compute_Tsurf_FD(GreensFunc,Pump_beam,XIz,skin_depth,int_axis):
    # Note that XIz must be using sparse = True.
    
    F_sens = 1/(1+skin_depth**2*XIz**2)# sensitivity function to integrate along z
    dTsurf = simpson(GreensFunc*F_sens*Pump_beam,XIz,axis=int_axis)/np.pi
    return dTsurf

    
    
# ----------------------------------- FDTR Signal Computing ----------------------------------#

def compute_Homega_1D(GreensFunc,Pump_beam,Sens_beam,XIr):
    # compute pump probe thermal response in frequency domain, in cylindrical coordinate

    #print(F_sens.shape)

    dT0 = GreensFunc*Pump_beam*Sens_beam*XIr
    
    Xir = XIr[:,0] # cast to 2D meshgrid.
    
    Hw = simpson(dT0*np.pi,Xir,axis=0)
    
    #phase = np.arctan(np.imag(Homega)/np.real(Homega))*180/np.pi
    
    return Hw



def compute_Homega(GreensFunc,Pump_beam,Sens_beam,XIr,XIz,skin_depths):
    # compute pump probe thermal response in frequency domain, in cylindrical coordinate
    d0 = skin_depths[0]
    d1 = skin_depths[1]
    f0 = 1/(1+d0**2*XIz**2)# sensitivity function to integrate along z
    f1 = 1/(1+d1**2*XIz**2)# sensitivity function to integrate along z
    F_sens = f0*f1
    #print(F_sens.shape)

    
    dT0 = simpson(GreensFunc*F_sens*Pump_beam*Sens_beam*XIr,XIz,axis=1)*2/np.pi # surface temperature rise
    Xir = XIr[:,0,0] # cast to 2D meshgrid.
    
    Hw = simpson(dT0*np.pi,Xir,axis=0)
    
    #phase = np.arctan(np.imag(Homega)/np.real(Homega))*180/np.pi
    
    return Hw

def compute_Homega_2Dxy(GreensFunc,Pump_beam,Sens_beam,XIx,XIy):
    # compute pump probe thermal response in frequency domain, in cylindrical coordinate

    #print(F_sens.shape)
    
    GTxy = GreensFunc*Pump_beam*Sens_beam

    #print(GTxy.shape)
    dTx = simpson(GTxy,XIy[0,:,0],axis=1)
    #print(dTx.shape)
    Hw = simpson(dTx,XIx[:,0,0],axis=0)
    
    return Hw


def compute_Homega_3D(GreensFunc,Pump_beam,Sens_beam,XIx,XIy,XIz,skin_depths):
    # compute pump probe thermal response in frequency domain, in cylindrical coordinate
    d0 = skin_depths[0]
    d1 = skin_depths[1]
    f0 = 1/(1+d0**2*XIz**2)# sensitivity function to integrate along z
    f1 = 1/(1+d1**2*XIz**2)# sensitivity function to integrate along z
    F_sens = f0*f1
    #print(F_sens.shape)

    
    GTxy = simpson(GreensFunc*F_sens*Pump_beam*Sens_beam,XIz,axis=2) # surface temperature rise
    #print(GTxy.shape)
    dTx = simpson(GTxy,XIy[0,:,0,0],axis=1)
    #print(dTx.shape)
    Hw = simpson(dTx,XIx[:,0,0,0],axis=0)
    
    return Hw


def calc_donut_FDTRSig(rd,rs,XIr,XIz,GreensFunc,skin_depths):
    # OmegaH doesn't need to be in the input. But one should input XIr,XIz by:
    #      XIr, XIz, OMEGAH =np.meshgrid(Xir,Xiz,OmegaH_Trads,sparse=True , indexing='ij')
    #
    
    P_donut = np.exp(-XIr**2*rd**2/4)*(1-XIr**2*rd**2/4)
    S_gauss = Gauss_Hankel(XIr,rs)
    
    Homega = compute_Homega(GreensFunc,P_donut,S_gauss,XIr,XIz,skin_depths)
    
    phase = np.arctan(np.imag(Homega)/np.real(Homega))*180/np.pi
    
    #smooth the phase.
    dphase = np.abs(np.diff(phase))
    if np.max(dphase)>120:
        loc = np.where(dphase == np.max(dphase))[0][0]
        phase[loc+1:]-=180
        
    return Homega,phase


def calc_ring_FDTRSig_2Dxy(R0,rp,rs,XIx,XIy,GreensFunc,offset_xy=(0,0)):
    
    # R0 is the ring radius, rp and rs are 1/e^2 half-widths of the beam.
    
    P_ring = Ring_Fourier(XIx,XIy, R0, rp) 
    # P_ring = RectRing_Hankel(XIr, R0, rp) #np.exp(-XIr**2*rp**2/8)*j0(XIr*R0) 
    
    # https://arxiv.org/pdf/1612.02665, The Journal of the Acoustical Society of America 140.4 (2016): 2829-2838.
    S_gauss =  Gauss_Offset_2DFT(XIx, XIy, rs, offset_xy)
    
    Homega = compute_Homega_2Dxy(GreensFunc, P_ring, S_gauss, XIx,XIy)
    
    phase = np.arctan(np.imag(Homega)/np.real(Homega))*180/np.pi
    
    #smooth the phase.
    
    dphase = np.abs(np.diff(phase))
    if np.max(dphase)>120:
        loc = np.where(dphase == np.max(dphase))[0][0]
        phase[loc+1:]-=180
            
        # ns +=1
        
    return Homega,phase  


def calc_ring_FDTRSig(R0,rp,rs,XIr,XIz,GreensFunc,skin_depths,offset=0.0,Nmax=50):
    
    # R0 is the ring radius, rp and rs are 1/e^2 half-widths of the beam.
    
    P_ring = Ring_Hankel(XIr, R0, rp) #RectRing_Hankel(XIr, R0, rp) #np.exp(-XIr**2*rp**2/8)*j0(XIr*R0) 
    
    # https://arxiv.org/pdf/1612.02665, The Journal of the Acoustical Society of America 140.4 (2016): 2829-2838.
    S_gauss = SensBeam_Hankel(XIr,rs,offset,Nmax) #Gauss_Hankel(XIr,rs)
    
    Homega = compute_Homega(GreensFunc,P_ring,S_gauss,XIr,XIz,skin_depths)
    
    phase = np.arctan(np.imag(Homega)/np.real(Homega))*180/np.pi
    
    #smooth the phase.
    
    dphase = np.abs(np.diff(phase))
    if np.max(dphase)>120:
        loc = np.where(dphase == np.max(dphase))[0][0]
        phase[loc+1:]-=180
            
        # ns +=1
        
    return Homega,phase    




def calc_FDTRSig(offset,rp,rs,XIr,XIz,GreensFunc,skin_depths,Nmax=40):
    # OmegaH doesn't need to be in the input. But one should input XIr,XIz by:
    #      XIr, XIz, OMEGAH =np.meshgrid(Xir,Xiz,OmegaH_Trads,sparse=True , indexing='ij')
    #
    P_gauss = Gauss_Hankel(XIr,rp) #np.exp(-(XIr*rp)**2/8)
    if np.abs(offset) < rp*0.001:
        S_gauss = Gauss_Hankel(XIr,rs)
    else:
        S_gauss = SensBeam_Hankel(XIr,rs,offset,Nmax) #np.exp(-(XIr*rp)**2/8)
    
    Homega = compute_Homega(GreensFunc,P_gauss,S_gauss,XIr,XIz,skin_depths)
    
    phase = np.arctan(np.imag(Homega)/np.real(Homega))*180/np.pi
    
    #smooth the phase.
    dphase = np.abs(np.diff(phase))
    if np.max(dphase)>120:
        loc = np.where(dphase == np.max(dphase))[0][0]
        phase[loc+1:]-=180

        
    return Homega,phase
    
def calc_Offset_FDTRSig(offsets,rp,rs,XIr,XIz,GreensFunc,skin_depths,Nmax=40):
    
    Hws = []
    phases = []
    
    for i,offset in enumerate(offsets):
        Hw,phase = calc_FDTRSig(offset,rp,rs,XIr,XIz,GreensFunc,skin_depths,Nmax)        
        Hws.append(Hw)
        phases.append(phase)
        
    Hws = np.array(Hws)
    
    
    Noffsets,Nfreqs = Hws.shape
    
    mags = np.max(np.abs(Hws),axis=0)
    mags_repmat = np.broadcast_to(mags,(Noffsets,Nfreqs))
        
        
    return Hws,Hws/mags_repmat,np.array(phases)


def calc_FDTRSig_3D(offset_xy,rp,rs,XIx,XIy,XIz,GreensFunc,skin_depths):
    # OmegaH doesn't need to be in the input. But one should input XIr,XIz by:
    #      XIr, XIz, OMEGAH =np.meshgrid(Xir,Xiz,OmegaH_Trads,sparse=True , indexing='ij')
    #
    
    
    P_gauss = Gauss_2DFT(XIx,XIy,rp)    
    S_gauss = Gauss_Offset_2DFT(XIx, XIy, rs, offset_xy)

    
    Homega = compute_Homega_3D(GreensFunc,P_gauss,S_gauss,XIx,XIy,XIz,skin_depths)
    
    phase = np.arctan(np.imag(Homega)/np.real(Homega))*180/np.pi
    
    #smooth the phase.
    dphase = np.diff(phase)
    if np.max(dphase)>100:
        loc = np.where(dphase == np.max(dphase))[0][0]
        phase[loc+1:]-=180
    
    return Homega,phase

def calc_Ring_FDTRSig_3D(R0,rp,rs,XIx,XIy,XIz,GreensFunc,skin_depths):
    
    P_ring = RectRing_Fourier(XIx, XIy, R0, rp)
    S_gauss = Gauss_2DFT(XIx,XIy,rs) #Gauss_Offset_2DFT(XIx, XIy, rs, offset_xy)

    
    Homega = compute_Homega_3D(GreensFunc,P_ring,S_gauss,XIx,XIy,XIz,skin_depths)
    
    phase = np.arctan(np.imag(Homega)/np.real(Homega))*180/np.pi
    
    #smooth the phase.
    dphase = np.diff(phase)
    if np.max(np.abs(dphase))>100:
        loc = np.where(dphase == np.max(dphase))[0][0]
        phase[loc+1:]-=180
    
    return Homega,phase



def calc_Offset_FDTRSig_3D(offsets,rp,rs,XIx,XIy,XIz,GreensFunc,skin_depths):
    
    Hws = []
    phases = []
    
    for i,offset in enumerate(offsets):
        Hw,phase = calc_FDTRSig_3D(offset,rp,rs,XIx,XIy,XIz,GreensFunc,skin_depths)        
        Hws.append(Hw)
        phases.append(phase)
        
    Hws = np.array(Hws)
    
    
    Noffsets,Nfreqs = Hws.shape
    
    mags = np.max(np.abs(Hws),axis=0)
    mags_repmat = np.broadcast_to(mags,(Noffsets,Nfreqs))
                
    return Hws,Hws/mags_repmat,np.array(phases)



# ----------------------------------- TTR Signal Computing ----------------------------------#

def calc_TTRSig_cyln(rp,rs,XIr,XIz,OmegaH_Trads,GreensFunc,skin_depths,Tmax,Wmax,method='PCI',offset=0,Nmax=40):
    Ws = np.pi/Tmax
    #Homega,phase = calc_donut_FDTRSig(rp,rs,XIr,XIz,GreensFunc,skin_depths) 
    Homega,phase = calc_FDTRSig(offset,rp,rs,XIr,XIz,GreensFunc,skin_depths,Nmax=Nmax)
    
    omega_intp,Fwp_intp = Interp_FDsignal(OmegaH_Trads,Homega,Ws,Wmax,method)
    t,Tt = InvFourTrans_Even(omega_intp,Fwp_intp,Ws)
    
    return t,Tt



def calc_TTRSig_3D(rp,rs,XIx,XIy,XIz,OmegaH_Trads,GreensFunc,skin_depths,Tmax,Wmax,method='PCI',offset=(0,0)):
    Ws = np.pi/Tmax
    Homega,phase = calc_FDTRSig_3D(offset,rp,rs,XIx,XIy,XIz,GreensFunc,skin_depths)
    omega_intp,Fwp_intp = Interp_FDsignal(OmegaH_Trads,Homega,Ws,Wmax,method)
    t,Tt = InvFourTrans_Even(omega_intp,Fwp_intp,Ws)
    if Tt[0]<0:
        Tt -= np.real(simpson(Fwp_intp,omega_intp))/np.pi   
    return t,Tt

 
def calc_ring_TTRSig_2Dxy(R0,rp,rs,XIx,XIy,OmegaH_Trads,GreensFunc,Tmax,Wmax,tstep_ps = 25.0,method='PCI',offset_xy=(0,0)):
    Ws = np.pi/Tmax
    Homega,phase = calc_ring_FDTRSig_2Dxy(R0, rp, rs, XIx, XIy, GreensFunc,offset_xy)
    
    # OmegaH_Trads = np.append(0,OmegaH_Trads)
    # Homega = np.append(np.real(Homega[0]),Homega)
    omega_intp,Fwp_intp = Interp_FDsignal(OmegaH_Trads,Homega,Ws,Wmax,method)
    #Regulizer = 1/(a**2+omega_intp**2)
    
    t,Tt = InvFourTrans_Hermit(omega_intp,Fwp_intp,Ws)
        
    pci = PchipInterpolator(t,Tt)
    
    ts = np.arange(np.min(t),np.max(t),tstep_ps)
    
    T_ts = pci(ts)
    
    if T_ts[0] <0:
        
        T0 = np.real(simpson(Fwp_intp,omega_intp))/np.pi
        T_ts -= T0
     
    
    return ts,T_ts



def calc_ring_TTRSig_3D(R0,rp,rs,XIx,XIy,XIz,OmegaH_Trads,GreensFunc,skin_depths,Tmax,Wmax, tstep_ps = 25.0,method='PCI'):
    Ws = np.pi/Tmax
    Homega,phase = calc_Ring_FDTRSig_3D(R0,rp,rs,XIx,XIy,XIz,GreensFunc,skin_depths)
    omega_intp,Fwp_intp = Interp_FDsignal(OmegaH_Trads,Homega,Ws,Wmax,method)
    t,Tt = InvFourTrans_Even(omega_intp,Fwp_intp,Ws)
    if Tt[0]<0:
        Tt -= np.real(simpson(Fwp_intp,omega_intp))/np.pi  
    pci = PchipInterpolator(t,Tt)
    
    ts = np.arange(np.min(t),np.max(t),tstep_ps)
    T_ts = pci(ts)
    
    return ts,T_ts
    

def calc_Temperature_xyt_2D(xy_pos,t0,R0,rp,rs,XIx,XIy,OmegaH_Trads,GreensFunc,Tmax,Wmax,method_intp='PCI'):
    # specify a time (ps), and calculate the temperature at the specific position.
    t,Tt_xy = calc_ring_TTRSig_2Dxy(R0,rp,rs,XIx,XIy,OmegaH_Trads,GreensFunc,Tmax,Wmax,method = method_intp,offset_xy=xy_pos)
    
    tps_cut = 5*t0
    
    tc = t[t<tps_cut]
    Tt_xy = Tt_xy[t<tps_cut]
    
    # by setting rs =0, we are essentially calculating inverse Fourier transform.
    
    pci = PchipInterpolator(tc,Tt_xy)
    Tt0_xy = pci(t0)
    
    return Tt0_xy
    


# ----------------------------------- TTG Signal Computing ----------------------------------#
def calc_TTGSig_skindepth(XIx,XIz,OmegaH_Trads,GreensFunc,skin_depths,Tmax,Wmax,method='PCI'):
    
    '''
     This function calcualtes temperature varations in time domain of TTG signal, with
     skin depths of pump and probe beams considered.
    '''
    Ws = np.pi/Tmax
    
    THz = 1e12
    
    Greens_Func_cosqx = np.sum(GreensFunc,axis=0)*np.pi*THz
    
    d0 = skin_depths[0]
    d1 = skin_depths[1]
    
    f0 = 1/(1+d0**2*XIz**2)# sensitivity function to integrate along z
    f1 = 1/(1+d1**2*XIz**2)# sensitivity function to integrate along z
    F_sens = f0*f1
    
    dT0 = simpson(Greens_Func_cosqx*F_sens,XIz,axis=1)/np.pi/2

    omega_intp,dT_intp = Interp_FDsignal(OmegaH_Trads,dT0[0],Ws,Wmax,method)
    t,Tt = InvFourTrans_Even(omega_intp,dT_intp,Ws)
    
    return t,Tt,omega_intp,dT_intp


def calc_TTGSig(XIx,OmegaH_Trads,GreensFunc,Tmax,Wmax,method='PCI'):
    Ws = np.pi/Tmax
    
    THz = 1e12
    
    dT0 = np.sum(GreensFunc,axis=0)*np.pi*THz

    omega_intp,dT_intp = Interp_FDsignal(OmegaH_Trads,dT0,Ws,Wmax,method)
    t,Tt = InvFourTrans_Even(omega_intp,dT_intp,Ws)
    

    return t,Tt,omega_intp,dT_intp


# ----------------------------------- TDTR Signal Computing ----------------------------------#

def calc_TDTRSig(td_ns_model,OmegaH_Trads,offset,rp,rs,XIr,XIz,GreensFunc,skin_depths,Nmax=40,if_pump_adv=False):
    Td,WH = np.meshgrid(td_ns_model,OmegaH_Trads)
    exp_jwd = np.exp(1j*WH*Td*1000)
    
    Hw,phase = calc_FDTRSig(offset,rp,rs,XIr,XIz,GreensFunc,skin_depths,Nmax)
    
    Zd = np.tensordot(Hw,exp_jwd,axes=1) # pump delayed signal
    omega0 = np.min(np.abs(OmegaH_Trads)) # modulation frequency in THz
    if type(td_ns_model) == list or tuple or float:
        td_ns_model = np.array(td_ns_model)
    
    extraphase = np.exp(1j*omega0*td_ns_model*1000*if_pump_adv)
    Z = Zd*extraphase
    phase = np.arctan(np.imag(Z )/np.real(Z))*180/np.pi
    # smooth the phase.
    dphase = np.diff(phase)
    if np.max(dphase)>120:
        loc = np.where(dphase == np.max(dphase))[0][0]
        phase[loc+1:]-=180
    
    return Z,phase
    
    
def calc_Offset_TDTRSig(td_ns_model,OmegaH_Trads,offsets,rp,rs,XIr,XIz,GreensFunc,skin_depths,Nmax=40,if_pump_adv=False):
    
    Zs = []
    Ratios = []
    
    for i,offset in enumerate(offsets):
        Z,Ratio = calc_TDTRSig(td_ns_model,OmegaH_Trads,offset,rp,rs,XIr,XIz,GreensFunc,skin_depths,Nmax,if_pump_adv)     
        Zs.append(Z)
        Ratios.append(Ratio)
        
    Zs = np.array(Zs)
    
    Noffsets,Ntds = Zs.shape
    
    mags = np.max(np.abs(Zs),axis=0)
    mags_repmat = np.broadcast_to(mags,(Noffsets,Ntds))
                
    return Zs,Zs/mags_repmat,np.array(Ratios)



def calc_TDTRSig_3D(td_ns_model,OmegaH_Trads,offset,rp,rs,XIx,XIy,XIz,GreensFunc,skin_depths,Nmax=40,if_pump_adv=False):
    Td,WH = np.meshgrid(td_ns_model,OmegaH_Trads)
    exp_jwd = np.exp(1j*WH*Td*1000)
    
    Hw,phase = calc_FDTRSig(offset,rp,rs,XIx,XIy,XIz,GreensFunc,skin_depths,Nmax)
    
    Zd = np.tensordot(Hw,exp_jwd,axes=1) # pump delayed signal
    omega0 = np.min(np.abs(OmegaH_Trads)) # modulation frequency in THz
    if type(td_ns_model) == list or tuple or float:
        td_ns_model = np.array(td_ns_model)
    
    extraphase = np.exp(1j*omega0*td_ns_model*1000*if_pump_adv)
    Z = Zd*extraphase
    phase = np.arctan(np.imag(Z )/np.real(Z))*180/np.pi
    # smooth the phase.
    dphase = np.diff(phase)
    if np.max(dphase)>120:
        loc = np.where(dphase == np.max(dphase))[0][0]
        phase[loc+1:]-=180
    
    return Z,phase
    
    

def calc_Offset_TDTRSig_3D(td_ns_model,OmegaH_Trads,offsets,rp,rs,XIx,XIy,XIz,GreensFunc,skin_depths,Nmax=40,if_pump_adv=False):
    
    Zs = []
    Ratios = []
    
    for i,offset in enumerate(offsets):
        Z,Ratio = calc_TDTRSig(td_ns_model,OmegaH_Trads,offset,rp,rs,XIx,XIy,XIz,GreensFunc,skin_depths,Nmax,if_pump_adv)     
        Zs.append(Z)
        Ratios.append(Ratio)
        
    Zs = np.array(Zs)
    
    Noffsets,Ntds = Zs.shape
    
    mags = np.max(np.abs(Zs),axis=0)
    mags_repmat = np.broadcast_to(mags,(Noffsets,Ntds))
        
        
    return Zs,Zs/mags_repmat,np.array(Ratios)
