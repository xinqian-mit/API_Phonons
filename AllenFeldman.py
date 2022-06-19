import numpy as np
from phonopy.units import VaspToTHz,AMU,EV
from phonopy.harmonic.force_constants import similarity_transformation
from numba import jit,njit,prange
from API_phonopy import phonopyAtoms_to_aseAtoms,aseAtoms_to_phonopyAtoms

def get_Va_ijbc(phonon):
    """
    obtain the velocity operator
    (Va)_ij^bc = (Ria-Rja)*\Phi_ij^bc/sqrt(Mi*Mj)
    """
    atoms = phonopyAtoms_to_aseAtoms(phonon.get_primitive()) # supercell and primitive cell should be the same.
    masses = atoms.get_masses()
    Natom = atoms.get_global_number_of_atoms()
    
    Rvec_atoms = np.zeros((Natom,Natom,3))
    
    for i in range(Natom):
        for j in range(i):
            xij,yij,zij,rij = find_nearest(atoms,i,j)
            Rvec_atoms[i,j] = np.array([xij,yij,zij])
            
    Phi = phonon.get_force_constants() # get force constants.
    
    Vx = np.zeros((Natom,Natom,3,3))
    Vy = np.zeros((Natom,Natom,3,3))
    Vz = np.zeros((Natom,Natom,3,3))
    
    for i in range(Natom):
        for j in range(i):
            Vx_ij = -Rvec_atoms[i,j,0]*Phi[i,j]/np.sqrt(masses[i]*masses[j])
            Vy_ij = -Rvec_atoms[i,j,1]*Phi[i,j]/np.sqrt(masses[i]*masses[j])
            Vz_ij = -Rvec_atoms[i,j,2]*Phi[i,j]/np.sqrt(masses[i]*masses[j])
            
            Vx[i,j]=Vx_ij
            Vx[j,i]=-Vx_ij.T
            Vy[i,j]=Vy_ij
            Vy[j,i]=-Vy_ij.T         
            Vz[i,j]=Vz_ij
            Vz[j,i]=-Vz_ij.T
        
    # flatten the velocity operator
    # Va_(i,j,b,c) to Va_[(ib),(jc)]
    
    Vx_flat = np.reshape(Vx.transpose(0,2,1,3),(Natom*3,Natom*3))
    Vy_flat = np.reshape(Vy.transpose(0,2,1,3),(Natom*3,Natom*3))
    Vz_flat = np.reshape(Vz.transpose(0,2,1,3),(Natom*3,Natom*3))
    
    return Vx_flat,Vy_flat,Vz_flat


def get_velmat_modepairs(freqs,eigvecs,Vx_flat,Vy_flat,Vz_flat,factor=VaspToTHz):
    """
       va_mn = em*(Va*en) computes the pairwise velocity operators.
       for phonon gas model, va_mn = va_n*\delta_mn
    """
    freqs = np.reshape(np.abs(freqs),(1,len(freqs))) # some very small negative frequencies
    sqrt_fnfm = np.sqrt(freqs.T*freqs)
    temp_vx = np.dot(Vx_flat,eigvecs)
    vx_modepairs = np.dot(eigvecs.T,temp_vx)/sqrt_fnfm/2*factor**2 # ATHz
    
    temp_vy = np.dot(Vy_flat,eigvecs)
    vy_modepairs = np.dot(eigvecs.T,temp_vy)/sqrt_fnfm/2*factor**2 # ATHz
    
    temp_vz = np.dot(Vz_flat,eigvecs)
    vz_modepairs = np.dot(eigvecs.T,temp_vz)/sqrt_fnfm/2*factor**2 # ATHz
    
    return vx_modepairs,vy_modepairs,vz_modepairs


    
def find_nearest(atoms,i,j): # under periodic condition, find the true distance
    #for 3-dim case

    distance=atoms.get_distance(i,j,vector=True)
    xdc=distance[0]
    ydc=distance[1]
    zdc=distance[2]

    rmin=10000.0

    rv=atoms.cell

    if (atoms.cell.shape[0] ==3):
        #set
        xcdi=xdc-2.0*rv[0,0]
        ycdi=ydc-2.0*rv[1,0]
        zcdi=zdc-2.0*rv[2,0]

        for ii in [-1,0,1]:
            xcdi=xcdi+rv[0,0]
            ycdi=ycdi+rv[1,0]
            zcdi=zcdi+rv[2,0]

            xcdj=xcdi-2.0*rv[0,1]
            ycdj=ycdi-2.0*rv[1,1]
            zcdj=zcdi-2.0*rv[2,1]

            for jj in [-1,0,1]:
                xcdj=xcdj+rv[0,1]
                ycdj=ycdj+rv[1,1]
                zcdj=zcdj+rv[2,1]
                xcrd = xcdj - 2.0*rv[0,2]
                ycrd = ycdj - 2.0*rv[1,2]
                zcrd = zcdj - 2.0*rv[2,2]

                for kk in [-1,0,1]:
                    xcrd = xcrd + rv[0,2]
                    ycrd = ycrd + rv[1,2]
                    zcrd = zcrd + rv[2,2]
                    r = xcrd*xcrd + ycrd*ycrd + zcrd*zcrd
                    if (r<rmin):
                        rmin = r
                        xdc = xcrd
                        ydc = ycrd
                        zdc = zcrd
    
    return xdc,ydc,zdc,rmin    



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
    
# Here, the linewidth is set at a fixed number. If one use the true linewidths, we get Dornadio's model.    
@njit(parallel=True)
def calc_Diff(freqs,vx_modepairs,vy_modepairs,vz_modepairs,LineWidth=1e-2,factor=VaspToTHz):
    A2m = 1e-10
    THz2Hz = 1e12
    

    Nmodes = len(freqs) #number of modes


    Diffusivity = np.zeros(Nmodes)
            
    
    for s in prange(Nmodes):
        Diff_s = 0.0
        ws = freqs[s]*2*np.pi
        for r in range(Nmodes):            
            wr = freqs[r]*2*np.pi
            wsr_avg = (ws+wr)/2.0
            tau_sr = double_lorentz(ws,wr,LineWidth,LineWidth) # THz^-1                                        
            Diff_s -= wsr_avg**2*(vx_modepairs[s,r]*vx_modepairs[r,s]+vy_modepairs[s,r]*vy_modepairs[r,s]+vz_modepairs[s,r]*vz_modepairs[r,s]).real*tau_sr/3.0 #A^2THz^2*THz-1 = A^2THz       
        Diffusivity[s] = Diff_s*(A2m**2*THz2Hz)/(ws**2) #A^2THz^4/THz^2*THz-1 = A^2THz
    
    
    return Diffusivity
    


    
