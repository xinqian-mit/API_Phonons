#from quippy.atoms import Atoms
import quippy
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import phonopy.interface.vasp as phonVasp
import phonopy.units as Units
from math import pi
import os, glob
import os.path
import shutil
import ase.io as io
from matscipy.neighbours import neighbour_list
#from quippy.farray import fzeros, frange, fidentity
#from quippy.units import BOHR, HARTREE, PI
import ase
import multiprocessing as mp
from joblib import Parallel, delayed
import copy as cp
from numba import njit 



## -------------------------------------- Convert atom object type between packages ----------------------------------------------#
def aseAtoms_to_phonopyAtoms(aseAtoms):
    return PhonopyAtoms(symbols=aseAtoms.get_chemical_symbols(),positions=aseAtoms.get_positions(),cell=aseAtoms.get_cell())

def phonopyAtoms_to_aseAtoms(PhonopyAtoms,pbc=[True,True,True]):
    aseAtoms = ase.Atoms(symbols=PhonopyAtoms.get_chemical_symbols(),positions=PhonopyAtoms.get_positions(),cell=PhonopyAtoms.get_cell())
    aseAtoms.set_pbc(pbc)
    Atomic_numbers = PhonopyAtoms.get_atomic_numbers()
    Atomic_type_tags = np.zeros(np.shape(Atomic_numbers))
    atomic_type_unique = np.unique(Atomic_numbers)
    for i,iZ in enumerate(atomic_type_unique):
        Atomic_type_tags[Atomic_numbers==iZ]=i
    aseAtoms.set_tags(Atomic_type_tags)
    aseAtoms.set_initial_charges()
    
    return aseAtoms


#def phonopyAtoms_to_quipAtoms(PhonScells_w_disps): newer verions of quippy uses ase atoms.
#    Scells_quippy=[];
#    for scells in PhonScells_w_disps:
#        quip_atoms=Atoms(cell=scells.get_cell(),positions=scells.get_positions())
#        quip_atoms.set_atomic_numbers(scells.get_atomic_numbers())
#       quip_atoms.set_masses(scells.get_masses())
#        Scells_quippy.append(quip_atoms)
#    return Scells_quippy

#def phonopyAtom_to_quipAtom(scell):    
#    quip_atom=Atoms(cell=scell.get_cell(),positions=scell.get_positions())
#    quip_atom.set_atomic_numbers(scell.get_atomic_numbers())
#    quip_atom.set_masses(scell.get_masses())
#    return quip_atom


def qpoints_Band_paths(HiSym_Qpoints,Nq_path):
    """
    This function takes  High Symetry Qpoints as input, generate qpoints along the BZ path.
    High symmetry points should be a 2D np array.
    """
    (Nq,DIM)=np.shape(HiSym_Qpoints)
    bands=[]
    if Nq>=1:
        for iq in np.arange(0,Nq-1):
            qstart=HiSym_Qpoints[iq]
            qend=HiSym_Qpoints[iq+1]
            band=[]
            for i in np.arange(0,Nq_path+1):
                band.append(np.array(qstart)+(np.array(qend)-np.array(qstart))/Nq_path*i)
            bands.append(band)
    if Nq==1:
        bands.append(HiSym_Qpoints)
    return bands

## -------------------------------------- Calculate Energy and Forces ----------------------------------------------#

    


def calc_force_sets_GAP(gp_xml_file,Scell_quippy):
    """
    This function is used to produce force constants from GAP potential.
    The list contains: 
    [[[f1x,f1y,f1z],[f2x,f2y,f2z],...,[fNx,fNy,fNz]]# forces of each atom in the 1st displaced supercell
     [[f1x,f1y,f1z],[f2x,f2y,f2z],...,[fNx,fNy,fNz]]# forces of each atom in the 2nd displaced supercell
     ...
    ]
    """
    energy_gap=np.array(0.0,order='F')
    force_gap=np.zeros(Scell_quippy[0].get_positions().transpose().shape,order='F')
    
    pot = quippy.potential.Potential(param_filename=gp_xml_file)
    
    force_gap_scells=[]
    for i,scell in enumerate(Scell_quippy):
        #print (np.sum(pot.get_energies(scell)))
        pot.calculate(scell,forces=force_gap)  # use calculate for new gap.
        force_gap_scells.append(np.array(force_gap).transpose().tolist())
    #print(force_gap_scells)
    return force_gap_scells

def Parallel_calc_force_sets_GAP(gp_xml_file,Scell_quippy):
    """
    This function is used to produce force constants from GAP potential.
    The list contains: 
    [[[f1x,f1y,f1z],[f2x,f2y,f2z],...,[fNx,fNy,fNz]]# forces of each atom in the 1st displaced supercell
     [[f1x,f1y,f1z],[f2x,f2y,f2z],...,[fNx,fNy,fNz]]# forces of each atom in the 2nd displaced supercell
     ...
    ]
    """
    #pool = mp.Pool(mp.cpu_count())
    #force_gap_scells=[]
    #force_gap_scells = [pool.apply(calc_force_GAP, args=(gp_xml_file,scell)) for scell in Scell_quippy]
    force_gap_scells = Parallel(n_jobs=mp.cpu_count())(delayed(calc_force_GAP)(gp_xml_file,scell) for scell in Scell_quippy) 
    return force_gap_scells 




def calc_energy_force_GAP(gp_xml_file,scell):
    #scell need to be a quippy object.
    energy_gap = np.array(0.0,order='F')
    force_gap=np.zeros(scell.get_positions().transpose().shape,order='F')
    pot = quippy.potential.Potential(param_filename=gp_xml_file)
    pot.calculate(scell,forces=force_gap)
    energy_gap = np.sum(pot.get_energies(scell))
    F_gap = np.array(force_gap).transpose().tolist()
    return energy_gap.tolist(),F_gap

def calc_force_GAP(gp_xml_file,scell):
    #scell need to be a quippy object.
    force_gap=np.zeros(scell.get_positions().transpose().shape,order='F')
    pot = quippy.potential.Potential(param_filename=gp_xml_file)
    pot.calculate(scell,forces=force_gap)
    F_gap = np.array(force_gap).transpose().tolist()
    return F_gap

def calc_force_quip(pot_flag,scell,file_pot=None,param_str=None):
    energy_gap = np.array(0.0,order='F')
    force_quip=np.zeros(scell.get_positions().transpose().shape,order='F')
    pot = quippy.potential.Potential(pot_flag,param_filename=file_pot,param_str=param_str)
    pot.calculate(scell,forces=force_quip)
    F_quip = np.array(force_quip).transpose().tolist()
    return F_quip

def calc_force_sets_quip(pot_flag,Scell_quippy,file_pot=None,param_str=None):
    """
    This function is used to produce force constants from GAP potential.
    The list contains: 
    [[[f1x,f1y,f1z],[f2x,f2y,f2z],...,[fNx,fNy,fNz]]# forces of each atom in the 1st displaced supercell
     [[f1x,f1y,f1z],[f2x,f2y,f2z],...,[fNx,fNy,fNz]]# forces of each atom in the 2nd displaced supercell
     ...
    ]
    """
    energy_quip=np.array(0.0,order='F')
    force_quip=np.array(np.zeros(Scell_quippy[0].get_positions().transpose().shape),order='F')
    
    pot = quippy.potential.Potential(pot_flag,param_filename=file_pot,param_str=param_str)
    
    force_quip_scells=[]
    for scell in Scell_quippy:
        pot.calculate(scell,forces=force_quip)
        force_quip_scells.append(np.array(force_quip).transpose().tolist())
    
    return force_quip_scells



def calc_energy_sets_GAP(gp_xml_file,Scell_quippy):
    """
    This function is used to produce force constants from GAP potential.
    The list contains: 
    [[[f1x,f1y,f1z],[f2x,f2y,f2z],...,[fNx,fNy,fNz]]# forces of each atom in the 1st displaced supercell
     [[f1x,f1y,f1z],[f2x,f2y,f2z],...,[fNx,fNy,fNz]]# forces of each atom in the 2nd displaced supercell
     ...
    ]
    """
    energy_gap=np.array(0.0)
    #force_gap=np.array(np.zeros(Scell_quippy[0].get_positions().transpose().shape),order='F')
    
    pot = quippy.potential.Potential(param_filename=gp_xml_file)
    
    energies=[]
    for scell in Scell_quippy:
        energy_gap=np.sum(pot.get_energies(scell))
        energies.append(energy_gap.tolist())
    
    return energies

## -------------------------------------- Calculate Eigenvectors and thermodisplacements ----------------------------------------------#                

def get_reshaped_eigvecs(phonon_scell):
    """
    This function takes a phonon object, and reshape the shape of eigvecs
    This function returns list of the array on the path
    On each path, the eigs are a 4D array [q_point_index_on_path][index_of_mode][index_of_atom][index_of_dim]
    
    If you conduct Latt Dynamics at Gamma point, use this one instead of get_reshaped_eigvecs_mesh()
    """
    eigvecs= phonon_scell._band_structure.get_eigenvectors() #phonon_scell.band_structure.eigenvectors
    (Npaths,Nqs_on_path,N_modes,Nvecelems)=np.shape(eigvecs)
    Natom = int(N_modes/3)
    eigvecs_new=[]
    eigvecs_new_on_path = np.zeros([Nqs_on_path,N_modes,Natom,3],dtype="c8")
    for eigvecs_on_path in eigvecs:
        for qpoint in range(Nqs_on_path):
            for s in range(N_modes):
                for i in range(Natom):
                    for m in range(3):
                        eigvecs_new_on_path[qpoint,s,i,m]=eigvecs_on_path[qpoint,i*3+m,s]
        eigvecs_new.append(eigvecs_new_on_path)
                        
    return eigvecs_new

def get_freq_reshaped_eigvec_atq(phonon_scell,q):
    frequencies,eigvecs = phonon_scell.get_frequencies_with_eigenvectors(q)
    (Natoms_x3,Nmodes) = np.shape(eigvecs)
    Natoms = int(Natoms_x3/3) #python3 is more rigorous on datatypes.
    #print(Natoms)
    eigvecs_new = np.zeros([Nmodes,Natoms,3],dtype="c8")
    for s in range(Nmodes):
        for i in range(Natoms):
            for m in range(3):
                eigvecs_new[s,i,m]=eigvecs[i*3+m,s]
    return frequencies,eigvecs_new

def get_reshaped_eigvecs_mesh(phonon_scell):
    """
    This should be used with set_band_structure in phononpy.
    Eigenvectors is a numpy array of three dimension.
    The first index runs through q-points.
    In the second and third indices, eigenvectors obtained
    using numpy.linalg.eigh are stored.

    The third index corresponds to the eigenvalue's index.
    The second index is for atoms [x1, y1, z1, x2, y2, z2, ...].
    """
    qpoints, weights, frequencies, eigvecs = phonon_scell.get_mesh()
    (Nqpoints,Natoms_x3,Nmodes)=np.shape(eigvecs)
    Natoms = int(Natoms_x3/3)
    eigvecs_new = np.zeros([Nqpoints,Nmodes,Natoms,3],dtype="c8")
    for iq in range(Nqpoints):
        for s in range(Nmodes):
            for i in range(Natoms):
                for m in range(3):
                    eigvecs_new[iq,s,i,m]=eigvecs[iq,i*3+m,s]
                    
    return eigvecs_new

def write_freq_velocity_qmesh(filename,phonon_scell):
    qpoints, weights, frequencies, eigvecs = phonon_scell.get_mesh()
    phonon_scell.set_group_velocity()
    fid=open(filename,'w')
    for iq,q in enumerate(qpoints):
        Vg_at_q = phonon_scell.get_group_velocity_at_q(q)
        for ibrch,freq in enumerate(frequencies[iq]):
            fid.write('{:3d} {:3d} {:3d}  {:6f}  {:6f}  {:6f}  {:6f}\n'.format(iq,ibrch,weights[iq],freq,Vg_at_q[ibrch][0],Vg_at_q[ibrch][1],Vg_at_q[ibrch][2]))
    fid.close()

def write_unitcell_eigvecs_qmesh_gulp(filename,eigvecs,phonon_scell):
    fid=open(filename,'w')
    qpoints, weights, frequencies, eigvecsraw = phonon_scell.get_mesh()
    Nqpoints, Nbranches, Nbasis, DIM =np.shape(eigvecs)
    UnitCell = phonon_scell.get_unitcell()
    
    Atomic_No = UnitCell.get_atomic_numbers()
    pos = UnitCell.get_positions()
    fid.write('{:6d}\n'.format(Nbasis))
    for i in range(Nbasis):
        atNo = Atomic_No[i]
        x = pos[i][0]
        y = pos[i][1]
        z = pos[i][2]
        fid.write('{:3d}       {:6f}       {:6f}       {:6f}\n'.format(atNo,x,y,z))
    
    fid.write('{:6d}\n'.format(Nqpoints))
    fid.write('{:6d}\n'.format(Nbranches))
    for iq in range(Nqpoints):
        qx = qpoints[iq][0]
        qy = qpoints[iq][1]
        qz = qpoints[iq][2]
        fid.write('K point at   {:6f}  {:6f}  {:6f} in BZ\n'.format(qx,qy,qz))
        
        if (qx == 0 and qy == 0 and qz == 0):
            formatstr = '{:6f}  {:6f}  {:6f}\n'
            for s in range(Nbranches):
                freq = frequencies[iq][s]
                fid.write('Mode{:7d}\n'.format(s+1))
                fid.write('     {:6f}\n'.format(freq))
                for i in range(Nbasis):
                    ex = np.real(eigvecs[iq][s][i][0])
                    ey = np.real(eigvecs[iq][s][i][1])
                    ez = np.real(eigvecs[iq][s][i][2])
                    fid.write(formatstr.format(ex,ey,ez))           
        else:
            formatstr = '{:6f}  {:6f}  {:6f}      {:6f}  {:6f}  {:6f}\n'
            for s in range(Nbranches):
                freq = frequencies[iq][s]
                fid.write('Mode{:7d}\n'.format(s+1))
                fid.write('     {:6f}\n'.format(freq))
                for i in range(Nbasis):
                    Rex = np.real(eigvecs[iq][s][i][0])
                    Rey = np.real(eigvecs[iq][s][i][1])
                    Rez = np.real(eigvecs[iq][s][i][2])
                    Iex = np.imag(eigvecs[iq][s][i][0])
                    Iey = np.imag(eigvecs[iq][s][i][1])
                    Iez = np.imag(eigvecs[iq][s][i][2])                    
                    fid.write(formatstr.format(Rex,Rey,Rez,Iex,Iey,Iez))
    fid.close()
        
        
            
def get_DFSETS_GAP(Scell0,Scell_snaps,gp_xml_file): 
    Nsnaps = len(Scell_snaps)
    pos0_frac = Scell0.get_scaled_positions()
    latt_vec = Scell0.get_cell()
    displacements = np.zeros([Nsnaps,Scell0.get_number_of_atoms(),3])
    forces = np.zeros([Nsnaps,Scell0.get_number_of_atoms(),3])
    for i,scell in  enumerate(Scell_snaps):
        #print(i)
        pos_frac = scell.get_scaled_positions()
        ur = pos_frac-pos0_frac
        ui = np.zeros(pos_frac.shape)
        fi = np.zeros(pos_frac.shape)
        for iat in range(scell.get_number_of_atoms()):
            for j in range(3): #periodic boundary condition, wrap the sscell vec
                ujr = ur[iat][j]
                if (np.abs(ujr)>np.abs(ujr+1)):
                    ur[iat][j] = ujr+1
                if (np.abs(ujr)>np.abs(ujr-1)):
                    ur[iat][j] = ujr-1 
            ui[iat][0]=ur[iat][0]*latt_vec[0][0]+ur[iat][1]*latt_vec[1][0]+ur[iat][2]*latt_vec[2][0] #get disps
            ui[iat][1]=ur[iat][0]*latt_vec[0][1]+ur[iat][1]*latt_vec[1][1]+ur[iat][2]*latt_vec[2][1]
            ui[iat][2]=ur[iat][0]*latt_vec[0][2]+ur[iat][1]*latt_vec[1][2]+ur[iat][2]*latt_vec[2][2]
        scell_qp = phonopyAtoms_to_aseAtoms(scell)
        fi = calc_force_GAP(gp_xml_file,scell_qp) # get forces
        displacements[i][:][:]=ui
        forces[i][:][:]=fi
    
    return displacements,forces
        
        
            
         
            
            
        
              
    
@njit         
def Bose_factor(T,freq_THz):
    if T==0.0:
        return 0.0    
    if freq_THz <= 0:
        freq_THz = np.amax(np.array([1.0e-6,np.abs(freq_THz)])) # the absolute value is to consider imaginary (negative) modes.
    exp_factor=np.exp(Units.Hbar*freq_THz*2.0*pi*1.0e12/(Units.Kb*T))
    n=1./(exp_factor-1.0)

    if exp_factor == np.inf:
        n = 0.0
    return n
    
@njit 
def calc_Amp_displacement(T,freq_THz,mass):
    if freq_THz <= 0:
        freq_THz = np.amax(np.array([1.0e-6,np.abs(freq_THz)]))
    n = Bose_factor(T,freq_THz)
    scale2m2 = Units.EV/(0.001/Units.Avogadro*2.0*pi*1.0e12)    
    Amp = np.sqrt(scale2m2*Units.Hbar*(2.0*n+1.)/2.0/mass/freq_THz)/Units.Angstrom
    if freq_THz < 0.5: # get rid of too large amplitudes...
        Amp = 0.0    
    return Amp
@njit     
def calc_Amp_displacement_classic(T,freq_THz,mass):
    kBT=Units.Kb*T*Units.EV # J = kg*m2/s2
    m = mass*0.001/Units.Avogadro # kg   
    omega = freq_THz*1.0e12*2.0*pi # 1/s
    Amp = np.sqrt(kBT/m)/omega/Units.Angstrom # in angstrom
    if freq_THz < 0.5:
        Amp = 0.0
    return Amp
    
def thermo_disp_along_eig(phonon_scell,T,Nsnapshots):  
    eigvecs = get_reshaped_eigvecs(phonon_scell) # eigvecs are reshaped.
    Eps_qpoints = eigvecs [0] # the first index is for list, the index for BZ-path
    Eps_array = Eps_qpoints[0]  # the second index is q-points on the BZ-path 
    # since we are calculating Gamma point of a supercell, the first two indices are 0
    
    Freqs = phonon_scell._band_structure.get_frequencies()#phonon_scell.band_structure.frequencies
    fTHz_qpoints = Freqs[0]
    frequencies = np.abs(fTHz_qpoints[0]) 
    
    masses = phonon_scell.get_supercell().get_masses()
    Natoms = phonon_scell.get_supercell().get_number_of_atoms()
    u_disps = np.zeros([Nsnapshots,Natoms,3],dtype="c8") # The displacements are real.
    #print u_disps
    #v_disps = np.zeros([Nsnapshots,Natoms,3],dtype="c8")
   
    for iconfig in range(Nsnapshots):
        #print(iconfig)
        for s,eps_s in enumerate(Eps_array): # eps_s is the eigvecs of mode s
            freq_THz = frequencies[s]
             # get rid of the translation modes that diverges...
            if s>2:
                for i,eps_si in enumerate(eps_s): # eps_si is the eigvec on atom i of the mode s
                    mass = masses[i]
                    #(i)
                    uis,vis = disp_atom_along_eigvec(T,freq_THz,mass,eps_si)
                    u_disps[iconfig][i] += uis
                    if np.linalg.norm(uis)>1:
                        print(i,s)          
    return u_disps.real

def Parallel_thermo_dispVel_along_eig(phonon_scell,T,Nsnapshots):
    """
    This function parallely generate snapshot with thermo displacements with velocites associated.
    thermo_disp_along_eig doesn't associated with velocity, and it's serial.
    """
    #pool = mp.Pool(mp.cpu_count())
    #force_gap_scells = [pool.apply(snapshot_along_eig, args=(phonon_scell,T)) for iconfig in range(Nsnapshots)]
    uivi = Parallel(n_jobs=mp.cpu_count())(delayed(snapshot_along_eig)(phonon_scell,T) for iconfig in range(Nsnapshots))
    uivi = np.array(uivi)
    ui = uivi[:,0,:,:]
    vi = uivi[:,1,:,:]
    return ui,vi
    
def Generate_Supercells_with_Disps(Scell_ph,u_disps,v_disps,paral = False): 
    # input & output are PhonopyAtom/ase atoms objects
    Nsnaps,Natoms,DIM = np.shape(u_disps)
    
    if paral:
        Scell_snaps = Parallel(n_jobs=mp.cpu_count())(delayed(Supercell_snap_with_disp)(Scell_ph,u_disps[iconfig],v_disps[iconfig]) for iconfig in range(Nsnaps))
    else:
        pos0 = Scell_ph.get_positions()
        Scell_snaps = []
        for isnap in range(Nsnaps):
            Scell_tmp = cp.deepcopy(Scell_ph)
            pos = pos0 + u_disps[isnap]
            Scell_tmp.set_positions(pos)
            Scell_tmp.set_velocities(v_disps[isnap])
            Scell_snaps.append(Scell_tmp)          

        
    return Scell_snaps

def Supercell_snap_with_disp(Scell_ph,u_disp,v_disp):
    pos0 = Scell_ph.get_positions()
    Scell_disp = cp.deepcopy(Scell_ph)
    pos = pos0+u_disp
    Scell_disp.set_positions(pos)
    Scell_disp.set_velocities(v_disp)
    return Scell_disp

def snapshot_along_eig(phonon_scell,T):
    eigvecs = get_reshaped_eigvecs(phonon_scell) # eigvecs are reshaped.
    Eps_qpoints = eigvecs [0] # the first index is for list, the index for BZ-path
    Eps_array = Eps_qpoints[0]  # the second index is q-points on the BZ-path 
    # since we are calculating Gamma point of a supercell, the first two indices are 0
    
    Freqs = phonon_scell._band_structure.get_frequencies()#phonon_scell.band_structure.frequencies
    fTHz_qpoints = Freqs[0]
    frequencies = np.abs(fTHz_qpoints[0]) 
    
    masses = phonon_scell.get_supercell().get_masses()
    Natoms = phonon_scell.get_supercell().get_number_of_atoms()
    u_disps = np.zeros([Natoms,3],dtype="c8")
    v_disps = np.zeros([Natoms,3],dtype="c8")
    
    for s,eps_s in enumerate(Eps_array): # eps_s is the eigvecs of mode s
        freq_THz = frequencies[s]
         # get rid of the translation modes that diverges...
        if s>2:
            for i,eps_si in enumerate(eps_s): # eps_si is the eigvec on atom i of the mode s
                mass = masses[i]
                uis,vis = disp_atom_along_eigvec(T,freq_THz,mass,eps_si)
                u_disps[i] += uis
                v_disps[i] += vis
                if np.linalg.norm(uis)>1:
                    print(i,s) 
    return u_disps.real,v_disps.real
    
@njit           
def disp_atom_along_eigvec(T,freq_THz,mass,eigvec):
    Amps_si = calc_Amp_displacement(T,freq_THz,mass)
    xi1 = np.random.random_sample()
    xi2 = np.random.random_sample()
    uis = eigvec*Amps_si*np.sqrt(-2.0*np.log(xi1))*np.sin(2.0*pi*xi2) # in Angstroms
    vis = (freq_THz*2.0*pi)*Amps_si*eigvec*np.sqrt(-2.0*np.log(xi1))*np.sin(2.0*pi*xi2) # in Angstrom/ps
    if np.linalg.norm(uis)>1:
        print(freq_THz,Amps_si,eigvec,np.sqrt(-2.0*np.log(xi1))*np.sin(2.0*pi*xi2))
    return uis,vis


def disp_atom_along_mode_qs(q_red,eigvec,Ncells,prim_cell): # here the time phase is not included here
    Nbasis = prim_cell.get_number_of_atoms()
    N = Nbasis*np.product(Ncells)
    U_disp = np.zeros([N,3],dtype="c8")
    masses = prim_cell.get_masses()
    iat = 0
    for ib in range(Nbasis):
        mb = masses[ib]*0.001/Units.Avogadro
        for lz in range(Ncells[2]):
            for ly in range(Ncells[1]):
                for lx in range(Ncells[0]):
                    phase = 1j*(lx*q_red[0]+ly*q_red[1]+lz*q_red[2])*2.0*pi
                    exp_phi = np.exp(phase)
                    U_disp[iat,:]=exp_phi*eigvec[ib]
                    iat+=1
    return U_disp


## -------------------------------------- File I/O ----------------------------------------------# 


    
    
def write_Supercells_VASP(Supercells,directory='./',prefix='POSCAR'):
    #print(directory)
    if directory!='./':
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)        
 
    
    if type(Supercells) != list:
        Supercells = [Supercells]
    
    NSnaps = len(Supercells)
    str_NSnaps = str(NSnaps)
    for isnap,supercell in enumerate(Supercells):
        str_isnap = str(isnap+1)
        index_len = np.max([3,len(str_NSnaps)])
        Nzero = index_len-len(str_isnap)
        index = ''
        for iz in range(Nzero):
            index += '0'
        index += str_isnap
        phonVasp.write_vasp(prefix+'-'+index,supercell)
        if directory!='./':
            shutil.move(prefix+'-'+index,directory)    
        
        
def write_xyz_aseAtomsList(AtomsList,filename):
    for at in AtomsList:
        ase.io.write(filename,at,format='xyz')

        
def write_2D_array(filename,data_2d):
    fid=open(filename,'w')
    dim,N = np.shape(data_2d)
    for i in range(N):
        fid.write('{:6f}  {:6f}\n'.format(data_2d[0][i],data_2d[1][i]))
    fid.close()

def write_ShengBTE_FC2(force_constants,
                          filename='FORCE_CONSTANTS_2ND',
                          p2s_map=None):
    """Write force constants in text file format.

    Parameters
    ----------
    force_constants: ndarray
        Force constants
        shape=(n_satom,n_satom,3,3) or (n_patom,n_satom,3,3)
        dtype=double
    filename: str
        Filename to be saved
    p2s_map: ndarray
        Primitive atom indices in supercell index system
        dtype=intc

    """

    if p2s_map is not None and len(p2s_map) == force_constants.shape[0]:
        indices = p2s_map
    else:
        indices = np.arange(force_constants.shape[0], dtype='intc')

    with open(filename, 'w') as w:
        fc_shape = force_constants.shape
        w.write("%4d %4d\n" % fc_shape[:2])
        for i, s_i in enumerate(indices):
            for j in range(fc_shape[1]):
                w.write("%4d%4d\n" % (s_i + 1, j + 1))
                for vec in force_constants[i][j]:
                    w.write(("%22.15f"*3 + "\n") % tuple(vec))

## -------------------------------------- Other Properties Calculation ----------------------------------------------# 
    
#def get_born_Charges(pot,at0, charge, dipole, dx=1e-5, args_str=None):  
#    at0.add_property('charge',charge)
#    at0.add_property('dipole',dipole)
#    fborn_tensor = fzeros((3,3,at0.n))
#    restart = True
#    for i in frange(at0.n):
#        for j in frange(3):
#            at = at0.copy()
#            at.pos[j,i] += -dx
#            pot.calc(at, force=True, restart=restart, args_str=args_str) # This helps to set the cutoff of atoms
#            at.calc_connect()
#            pot.calc(at, force=True, restart=restart, args_str=args_str)
#            dip1 = fzeros(3)
#            for k in frange(at.n):
#                dip1 += at.dipole[k] + at.charge[k]*at.pos[:,k]
#            at = at0.copy()
#            at.pos[j,i] += dx
#            pot.calc(at, force=True, restart=restart, args_str=args_str)
#            at.calc_connect()
        
        
#            dip2 = fzeros(3)
#            for k in frange(at.n):
#                dip2 += at.dipole[k] + at.charge[k]*at.pos[:,k]
        
#            fborn_tensor[:,j,i] = (dip2 - dip1)/(dx*2.0)
#    Z_born = np.zeros((at0.n,3,3))
#    for iat in range(at0.n):
#        for i in range(3):
#            for j in range(3):
#                Z_born[iat,i,j]=fborn_tensor[j+1,i+1,iat+1]
#    return Z_born
        

# calculate radial distribution function
def calc_rdf(Cell,rdf_cutoff,rdf_nbins):
    # use ase object for calculating rdf.
    r = neighbour_list('d', Cell, cutoff=rdf_cutoff)
    rdf, bin_edges = np.histogram(r, bins=rdf_nbins, range=(0, rdf_cutoff))
    # normalize by bin volume and total number of atoms
    rdf = rdf / (len(Cell) * 4*np.pi/3 * (bin_edges[1:]**3-bin_edges[:-1]**3))
    # normalize by cell volume
    rdf /= len(Cell)/Cell.get_volume()
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2    
    return bin_centers,rdf        
    
    
                                
                    

    
### Below this point is still under construction ###
## -------------------------------------- Other Miscellaneous Fucntions ----------------------------------------------# 
def get_SupercellIndex_i2lb(Ncells,Nbasis):
    """
    This returns a 4-element list, with the first three indices [lx, ly, lz] the 
    index of the supercell and the fourth index ib the index of basis atoms in the unit cell.
    """
    index_i2lb = []
    for ib in range(Nbasis):
        for lz in range(Ncells[2]):
            for ly in range(Ncells[1]):
                for lx in range(Ncells[0]):
                    index_i2lb.append([lx, ly, lz, ib])
    return np.array(index_i2lb)

def get_SupercellIndex_lb2i(lbvec,Ncells,Nbasis):
    """
    This function takes lvvec=[lx,ly,lz,ib] as input and returns the reduced index i in the supercelll
    """
    lx = lbvec[0]
    ly = lbvec[1]
    lz = lbvec[2]
    b  = lbvec[3]
    return b*Ncells[0]*Ncells[1]*Ncells[2]+lz*(Ncells[0]*Ncells[1])+ly*Ncells[0]+lx
                    

    
    
    
