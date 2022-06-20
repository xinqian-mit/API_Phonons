#!/usr/bin/env python
# coding: utf-8

import numpy as np
import API_thirdorder as FC3
import thirdorder_core
import thirdorder_common
from phonopy import Phonopy
import phonopy.interface.vasp as Intf_vasp
from phonopy.structure.atoms import PhonopyAtoms
import phonopy.file_IO as PhonIO
import API_phonopy as api_ph # remember to set this module to python path
import multiprocessing as mp
import API_phonopy_lammps as api_pl
import os
from ase.calculators.lammpslib import LAMMPSlib

results = []
def callback_phipar(result):
    results.append(result)

def calc_phipart(i,e,nirred,ntot,p,cmds,sposcar,namepattern,atomtypes='atomic'):
    phi_i = np.zeros([3,ntot],dtype='float64')
    for n in range(4): 
        isign = (-1)**(n // 2)
        jsign = -(-1)**(n % 2)
        # Start numbering the files at 1 for aesthetic
        # reasons.
        number = nirred * n + i + 1 # the number doesn't follow the order of 1,2,3,... 
        dsposcar = FC3.normalize_SPOSCAR(FC3.move_two_atoms(sposcar, e[1], e[3], isign * thirdorder_common.H, e[0], e[2], jsign * thirdorder_common.H))
        filename = namepattern.format(number)
        FC3.write_POSCAR(dsposcar, filename)
        Scell = Intf_vasp.read_vasp(filename)
        #print(Scell)
        os.remove(filename)
        lammps_header=['units metal',
                   'atom_style '+atomtypes,
                   'atom_modify map array sort 0 0']
        scell = api_ph.phonopyAtoms_to_aseAtoms(Scell)
        
        lmp = LAMMPSlib(lmpcmds=cmds, log_file='log.'+str(i),lammps_header=lammps_header)
        #print(lmp)
        scell.set_calculator(lmp)
        force = scell.get_forces()
        os.remove('log.'+str(i))
        phi_i -= (isign * jsign * force[p, :].T) # put the result in a queue object, which will be retrieved by get
    return (i,phi_i)

if __name__ == "__main__":

    Nprocesses = 48
    Nrepeat = [1,1,1]
    nneigh = 1 # neighbor cutoff
    cmds = ["pair_style sw","pair_coeff * * Si.sw Si"]

    poscar = FC3.read_POSCAR(".","POSCAR_512_SWrelx")
    natoms = len(poscar["types"])
    symops = thirdorder_core.SymmetryOperations(poscar["lattvec"], poscar["types"], poscar["positions"].T, 1e-5) # symops is an obj.
    sposcar = FC3.gen_SPOSCAR(poscar, Nrepeat[0], Nrepeat[1], Nrepeat[2])
    ntot = natoms * np.prod(Nrepeat)
    dmin, nequi, shifts = FC3.calc_dists(sposcar)
    frange = FC3.calc_frange(poscar, sposcar, nneigh, dmin)
    print('rcut = '+str(frange)+' nm')


    wedge = thirdorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts,frange)
    list4 = wedge.build_list4()
    nirred = len(list4)
    nruns = 4 * nirred
    width = len(str(4 * (len(list4) + 1)))
    namepattern = "3RD.POSCAR.{{0:0{0}d}}".format(width)


    Scells = []
    phipart = np.zeros((3, nirred, ntot))
    p = FC3.build_unpermutation(sposcar)

    pool = mp.Pool(Nprocesses)

    for i, e in enumerate(list4):
        pool.apply_async(calc_phipart,args = (i,e,nirred,ntot,p,cmds,sposcar,namepattern,'charge'), callback = callback_phipar)
        #calc_phipart(i,e,nirred,ntot,p,cmds,sposcar,namepattern,'charge')    

    pool.close()
    pool.join()    

    for result in results:
        i = result[0]
        phipart[:,i,:] = result[1] 

    phipart /= (400. * thirdorder_common.H * thirdorder_common.H)
    phifull = thirdorder_core.reconstruct_ifcs(phipart, wedge, list4,poscar, sposcar)
    #print(phifull.shape)
    thirdorder_common.write_ifcs(phifull, poscar, sposcar, dmin, nequi, shifts, frange,"FORCE_CONSTANTS_3RD")




