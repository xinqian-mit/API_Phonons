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


#from quippy.farray import fzeros, frange, fidentity
#from quippy.units import BOHR, HARTREE, PI
import ase
import multiprocessing as mp
#from joblib import Parallel, delayed
import copy as cp
from numba import njit 






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
        
        #pot.calculate(scell,forces=force_gap)  # use calculate for new gap.
        #force_gap_scells.append(np.array(force_gap).transpose().tolist())
        scell.set_calculator(pot)
        force_gap_scells.append(scell.get_forces().tolist())
    #print(force_gap_scells)
    return force_gap_scells


def calc_energy_force_GAP(gp_xml_file,scell):
    #scell need to be a quippy object.
    #energy_gap = np.array(0.0,order='F')
    #force_gap=np.zeros(scell.get_positions().transpose().shape,order='F')
    pot = quippy.potential.Potential(param_filename=gp_xml_file)
    #pot.calculate(scell,forces=force_gap)
    scell.set_calculator(pot)
    F_gap = scell.get_forces()
    energy_gap = scell.get_potential_energy()
    #F_gap = np.array(force_gap).transpose().tolist()
    return energy_gap.tolist(),F_gap

def calc_force_GAP(gp_xml_file,scell):
    #scell need to be a quippy object.
    #force_gap=np.zeros(scell.get_positions().transpose().shape,order='F')
    pot = quippy.potential.Potential(param_filename=gp_xml_file)
    #pot.calculate(scell,forces=force_gap)
    scell.set_calculator(pot)
    F_gap = scell.get_forces().tolist()
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
        scell.set_calculator(pot)
        energy_gap = scell.get_potential_energy()
        energies.append(energy_gap)
        #energy_gap=np.sum(pot.get_energies(scell))
        #energies.append(energy_gap.tolist())
    
    return energies

        
        
            
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
        from API_phonopy import phonopyAtoms_to_aseAtoms
        scell_qp = phonopyAtoms_to_aseAtoms(scell)
        fi = calc_force_GAP(gp_xml_file,scell_qp) # get forces
        displacements[i][:][:]=ui
        forces[i][:][:]=fi
    
    return displacements,forces
        
        
            
         
            
            
        
              
    



  
        
        
def write_xyz_aseAtomsList(AtomsList,filename):
    for at in AtomsList:
        ase.io.write(filename,at,format='xyz')

        


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
        

    
    
    
                                
                    

    

                    

    
    
    