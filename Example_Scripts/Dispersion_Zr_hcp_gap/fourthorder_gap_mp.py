#!/usr/bin/env python3

# usage:
# # python3 fourthorder_gap_mp.py na nb nc cutoff( -i|nm ) n_processes "directory/gp_new.xml"
import numpy as np
import quippy
import API_thirdorder as FC3
import Fourthorder_core
from Fourthorder_common import *
from phonopy import Phonopy
import phonopy.interface.vasp as Intf_vasp
from phonopy.structure.atoms import PhonopyAtoms
import phonopy.file_IO as PhonIO
import API_phonopy as api_ph # remember to set this module to python path
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
    for n in range(8): 
        isign=(-1)**(n//4)
        jsign=(-1)**(n%4//2)
        ksign=(-1)**(n%2)
        # Start numbering the files at 1 for aesthetic
        # reasons.
        number=nirred * n + i + 1 # the number doesn't follow the order of 1,2,3,... 
        dsposcar=FC3.normalize_SPOSCAR(
                move_three_atoms(sposcar,
                               e[2],e[5],isign*H,
                               e[1],e[4],jsign*H,
                               e[0],e[3],ksign*H))
        filename = namepattern.format(number)
        FC3.write_POSCAR(dsposcar, filename)
        Scell = Intf_vasp.read_vasp(filename)
        os.remove(filename)
        #print number
        scell = api_ph.phonopyAtoms_to_aseAtoms(Scell)
        pot = quippy.potential.Potential(param_filename=gp_xml_file)
        scell.calc = pot
        force = scell.get_forces()
        
        
        phi_i -= (isign * jsign * ksign * force[p,:].T)
        # put the result in a queue object, which will be retrieved by get
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
    
    
    print("Reading POSCAR")
    poscar=FC3.read_POSCAR(".")
    natoms=len(poscar["types"])
    
    
    
    print("Analyzing the symmetries")
    symops=Fourthorder_core.SymmetryOperations(
        poscar["lattvec"],poscar["types"],
        poscar["positions"].T,SYMPREC)
    print("- Symmetry group {0} detected".format(symops.symbol))
    print("- {0} symmetry operations".format(symops.translations.shape[0]))
    print("Creating the supercell")
    sposcar=gen_SPOSCAR(poscar,na,nb,nc)
    ntot=natoms*na*nb*nc
    ngrid=[na,nb,nc]
    print("Computing all distances in the supercell")
    dmin,nequi,shifts=calc_dists(sposcar)
    
    if nneigh != None:
        frange = calc_frange(poscar, sposcar, nneigh, dmin)
    
    
    print("Looking for an irreducible set of fourth-order IFCs")
    wedge=Fourthorder_core.Wedge(poscar,sposcar,symops,dmin,
                                nequi,shifts,frange)
                                
    print("- {0} quartet equivalence classes found".format(wedge.nlist))
    list6=wedge.build_list4()
    nirred=len(list6)
  #  print np.shape(list6),list6
    nruns=8*nirred
    print("- {0} DFT runs are needed".format(nruns))
    
    print(sowblock)
    print("Writing undisplaced coordinates to 4TH.SPOSCAR")
    FC3.write_POSCAR(FC3.normalize_SPOSCAR(sposcar),"4TH.SPOSCAR")
    #  print "Write coordinations to xyz.txt"
    #  write_pos(sposcar,ngrid,natoms,"xyz.txt")
    #  print "Output cell+atom indices from supercell indices"
    #  id2ind(ngrid,natoms,"cellbasismap.txt")
    #  print "Output cell+atom indices for each quartet"
    #  write_indexcell(ngrid,poscar,sposcar,dmin,nequi,shifts,frange,"indexfull.txt")
    width=len(str(8*(len(list6)+1)))
    namepattern="4TH.POSCAR.{{0:0{0}d}}".format(width)
    print("Writing displaced coordinates to 4TH.POSCAR.*")
    
    p=FC3.build_unpermutation(sposcar)
    phipart=np.zeros((3,nirred,ntot))
    
    pool = mp.Pool(Nprocesses)
    
    for i,e in enumerate(list6):
        pool.apply_async(calc_phipart,args = (i,e,nirred,ntot,p,gp_xml_file,sposcar,namepattern), callback = callback_phipar)
        
    pool.close()
    pool.join()
    
    for result in results:
        i = result[0]
        phipart[:,i,:] = result[1]
    
    
    print("Computing an irreducible set of anharmonic force constants")

    phipart/=(8000.*H*H*H)
    print("Reconstructing the full array")
    phifull=Fourthorder_core.reconstruct_ifcs(phipart,wedge,list6,poscar,sposcar)
    print("Writing the constants to FORCE_CONSTANTS_4TH")
    write_ifcs(phifull,poscar,sposcar,dmin,nequi,shifts,frange,"FORCE_CONSTANTS_4TH") 

