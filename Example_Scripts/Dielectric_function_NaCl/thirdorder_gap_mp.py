#!/usr/bin/env python3
# coding: utf-8
# usage:
# python3 thirdorder_gap_mp.py na nb nc cutoff( -i|nm ) n_processes "directory/gp_new.xml"
import numpy as np
import API_thirdorder as FC3
import thirdorder_core
import thirdorder_common
from phonopy import Phonopy
import phonopy.interface.vasp as Intf_vasp
from phonopy.structure.atoms import PhonopyAtoms
import phonopy.file_IO as PhonIO
import API_quippy as api_q
import API_phonopy as api_ph
import os, glob
import os.path
import shutil
import sys
import multiprocessing as mp



results = []
def callback_phipar(result):
    results.append(result)
 
def calc_phipart(i,e,nirred,ntot,p,gp_xml_file,sposcar,namepattern):
    
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
        os.remove(filename)
        #print number
        Scell_quip = api_ph.phonopyAtoms_to_aseAtoms(Scell)
        force = np.array(api_q.calc_force_GAP(gp_xml_file,Scell_quip))  
        phi_i -= (isign * jsign * force[p, :].T) # put the result in a queue object, which will be retrieved by get
    #phipart[:,i,:] = phi_i
    return (i,phi_i)

    

if __name__ == "__main__":
    na, nb, nc = [int(i) for i in sys.argv[1:4]]
    if min(na, nb, nc) < 1:
        sys.exit("Error: na, nb and nc must be positive integers")    
    if sys.argv[4][0] == "-":
        try:
            nneigh = -int(sys.argv[4])
        except ValueError:
            sys.exit("Error: invalid cutoff")
        if nneigh == 0:
            sys.exit("Error: invalid cutoff")
    else:
        nneigh = None
        try:
            frange = float(sys.argv[4])
        except ValueError:
            sys.exit("Error: invalid cutoff")
        if frange == 0.:
            sys.exit("Error: invalid cutoff")
            
    Nprocesses = int(sys.argv[5])

    gp_xml_file=sys.argv[6]
    
    


    poscar = FC3.read_POSCAR(".")
    natoms = len(poscar["types"])
    symops = thirdorder_core.SymmetryOperations(poscar["lattvec"], poscar["types"], poscar["positions"].T, 1e-5) # symops is an obj.
    sposcar = FC3.gen_SPOSCAR(poscar, na, nb, nc)
    ntot = natoms * na * nb * nc
    dmin, nequi, shifts = FC3.calc_dists(sposcar)


    if nneigh != None:
        frange = FC3.calc_frange(poscar, sposcar, nneigh, dmin)
    
    print('frange = '+str(frange*10)+' A')
# looking for irreducible fc3
    wedge = thirdorder_core.Wedge(poscar, sposcar, symops, dmin, nequi, shifts,frange)
    list4 = wedge.build_list4()
    nirred = len(list4)
    nruns = 4 * nirred
    width = len(str(4 * (len(list4) + 1)))
    namepattern = "3RD.POSCAR.{{0:0{0}d}}".format(width)

# generate snapshots to calc FC3 and convert to quippy atom objects.
    
    phipart = np.zeros([3, nirred, ntot],dtype='float64')
    
    p = FC3.build_unpermutation(sposcar)
    
    pool = mp.Pool(Nprocesses)

    for i,e in enumerate(list4):
        pool.apply_async(calc_phipart,args = (i,e,nirred,ntot,p,gp_xml_file,sposcar,namepattern), callback = callback_phipar)
       
    pool.close()
    pool.join()
    
    for result in results:
        i = result[0]
        phipart[:,i,:] = result[1] 

        
    phipart /= (400. * thirdorder_common.H * thirdorder_common.H)
    phifull = thirdorder_core.reconstruct_ifcs(phipart, wedge, list4,poscar, sposcar)
    thirdorder_common.write_ifcs(phifull, poscar, sposcar, dmin, nequi, shifts, frange,"FORCE_CONSTANTS_3RD")
    
    

     

