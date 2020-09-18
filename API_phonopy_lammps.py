from ase.calculators.lammpslib import LAMMPSlib
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
import ase
import multiprocessing as mp
from joblib import Parallel, delayed  
import copy as cp
import API_quippy_phonopy_VASP as api_qpv

## ------------------------------------- Get Lammps box parameters -----------------------------------------------------##


def get_lmp_boxbounds(Scell):
    vec_lo = Scell.get_celldisp()
    xlo = vec_lo[0][0]; ylo = vec_lo[1][0]; zlo = vec_lo[2][0];

    [A,B,C] = Scell.get_cell()
    Ah = A/np.linalg.norm(A)
    Bh = B/np.linalg.norm(B)
    Ch = C/np.linalg.norm(C)
    AxB = np.cross(A,B)
    AxB_h = AxB/np.linalg.norm(AxB)

    ax = np.linalg.norm(A)
    bx = np.dot(B,Ah)
    by = np.linalg.norm(np.cross(Ah,B))
    cx = np.dot(C,Ah)
    cy = np.dot(B,C)-bx*cx/by
    cz = np.sqrt(np.dot(C,C)-cx*cx-cy*cy)

    [a,b,c,alpha,beta,gamma]=Scell.get_cell_lengths_and_angles()
    cos_alpha = np.cos(alpha/180*np.pi)
    cos_beta = np.cos(alpha/180*np.pi)
    cos_gamma = np.cos(alpha/180*np.pi)

    lx = a 
    xy = b*cos_gamma
    xz = c*cos_beta
    ly = np.sqrt(b*b-xy*xy)
    yz = (b*c*cos_alpha-xy*xz)/ly
    lz = np.sqrt(c*c-xz*xz-yz*yz)

    xhi = xlo + lx
    yhi = ylo + ly
    zhi = zlo + lz

    xlo_bound = xlo + np.min([0.0,xy,xz,xy+xz])
    xhi_bound = xhi + np.max([0.0,xy,xz,xy+xz])
    ylo_bound = ylo + np.min([0.0,yz])
    yhi_bound = yhi + np.min([0.0,yz])
    zlo_bound = zlo
    zhi_bound = zhi
    return xlo_bound,xhi_bound,ylo_bound,yhi_bound,zlo_bound,zhi_bound,xy,xz,yz



def calc_lmp_force_sets(cmds,Scells_ph,atomtypes='atomic',logfile='log.lammps'): 
    """
    This function uses ase and lammps' python API to calculate forces. Comment this funciton if it's not installed.
    In cmd, specifies the potential    
    Scells takes the list of perturbated supercells,  phonopyatom objs.
    
    """
    lammps_header=['units metal',
                   'atom_style '+atomtypes,
                   'atom_modify map array sort 0 0']

    if type(Scells_ph)!=list:
        Scells_ph = [Scells_ph]
    
    force_scells=[]
    for scell_ph in Scells_ph:
        lammps = LAMMPSlib(lmpcmds=cmds, log_file=logfile,lammps_header=lammps_header) # lammps obj has to be in the loop.
        scell = api_qpv.phonopyAtoms_to_aseAtoms(scell_ph)
        scell.set_calculator(lammps)
        forces = scell.get_forces()
        force_scells.append(forces.tolist())
    
    return force_scells


def calc_lmp_force(cmds,Scell_ph,atomtypes='atomic',logfile='log.lammps'): 
    """
    This function uses ase and lammps' python API to calculate forces. 
    In cmd, specifies the potential    
    Scells takes the list of perturbated supercells,  phonopyatom objs.
    
    """
    lammps_header=['units metal',
                   'atom_style '+atomtypes,
                   'atom_modify map array sort 0 0']



    lammps = LAMMPSlib(lmpcmds=cmds, log_file=logfile,lammps_header=lammps_header) # lammps obj has to be in the loop.
    scell = api_qpv.phonopyAtoms_to_aseAtoms(Scell_ph)
    scell.set_calculator(lammps)
    forces = scell.get_forces()
    
    return forces


def write_lmp_data(filename,SimCell,molID=[]):
    Masses_of_atypes = np.unique(SimCell.get_masses())
    Number_of_atom_types = len(Masses_of_atypes)
    Masses = SimCell.get_masses()
    Pos = SimCell.get_positions()

    fid = open(filename,'w')
    fid.write('LAMMPS data file. \n')
    fid.write('\n')
    fid.write('    {} atoms\n'.format(SimCell.get_global_number_of_atoms()))
    fid.write('\n')
    fid.write('   {} atom types\n'.format(len(np.unique(SimCell.get_atomic_numbers()))))
    fid.write('\n')
    xlo_bound,xhi_bound,ylo_bound,yhi_bound,zlo_bound,zhi_bound,xy,xz,yz = get_lmp_boxbounds(SimCell)
    fid.write('    {:9f}    {:9f} xlo xhi\n'.format(xlo_bound,xhi_bound))
    fid.write('    {:9f}    {:9f} ylo yhi\n'.format(ylo_bound,yhi_bound))
    fid.write('    {:9f}    {:9f} zlo zhi\n'.format(zlo_bound,zhi_bound))
    fid.write('{:6f} {:6f} {:6f} xy xz yz\n'.format(xy,xz,yz))
    fid.write('\n')
    fid.write('Masses\n')
    fid.write('\n')
    for atype in range(Number_of_atom_types):
        fid.write('   {}   {:4f}\n'.format(atype+1,Masses_of_atypes[atype]))
    fid.write('\n')
    
    
    if len(molID) ==0:
        fid.write('Atoms # charge\n') # use atomic_style full
        fid.write('\n')
        for iat in range(SimCell.get_global_number_of_atoms()):
            for atype in range(Number_of_atom_types):
                if Masses[iat] == Masses_of_atypes[atype]:
                    tag = atype+1
            fid.write('{}   {}  {:6f}    {:9f}    {:9f}     {:9f} \n'.format(iat+1,tag,0.0,Pos[iat][0],Pos[iat][1],Pos[iat][2]))
        fid.write('\n')
        
    else:
        fid.write('Atoms # full\n')
        fid.write('\n')
        for iat in range(SimCell.get_number_of_atoms()):
            for atype in range(Number_of_atom_types):
                if Masses[iat] == Masses_of_atypes[atype]:
                    tag = atype+1
            fid.write('{}   {}   {}  {:6f}    {:9f}    {:9f}     {:9f} \n'.format(iat+1,molID[iat],tag,0.0,Pos[iat][0],Pos[iat][1],Pos[iat][2]))
        fid.write('\n')   

    fid.close()    

def write_lmp_dump(filename,Cell_snaps):
    fid = open(filename,'w')
    if type(Cell_snaps) != list:
        Cell_snaps = [Cell_snaps]
        
    Nsnaps = len(Cell_snaps) # Cell_snaps should be a list.
    
    for isnap,cell in enumerate(Cell_snaps):
        Natoms = cell.get_number_of_atoms()
        types = cell.get_tags()
        fid.write('ITEM: TIMESTEP\n')
        fid.write('{}\n'.format(isnap))
        fid.write('ITEM: NUMBER OF ATOMS\n')
        fid.write('{}\n'.format(Natoms))
        fid.write('ITEM: BOX BOUNDS xy xz yz pp pp pp\n')
        xlo_bound,xhi_bound,ylo_bound,yhi_bound,zlo_bound,zhi_bound,xy,xz,yz = get_lmp_boxbounds(cell)
        fid.write('{:6f} {:6f} {:6f}\n'.format(xlo_bound,xhi_bound,xy))
        fid.write('{:6f} {:6f} {:6f}\n'.format(ylo_bound,yhi_bound,xz))
        fid.write('{:6f} {:6f} {:6f}\n'.format(zlo_bound,zhi_bound,yz))
        fid.write('ITEM: ATOMS id type x y z vx vy vz\n')
        pos = cell.get_positions()
        vel = cell.get_velocities()
        for iat in range(Natoms):
            atype = types[iat]+1
            fid.write('{} {} {:6f} {:6f} {:6f} {:6f} {:6f} {:6f}\n'.format(iat,atype,pos[iat][0],pos[iat][1],pos[iat][2],vel[iat][0],vel[iat][1],vel[iat][2]))
    fid.close()