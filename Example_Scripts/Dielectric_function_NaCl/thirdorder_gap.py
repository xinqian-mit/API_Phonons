#!/usr/bin/env python3
# coding: utf-8
# usage:
# thirdorder_gap na nb nc cutoff( -i|nm ) "directory/gp_new.xml"
import numpy as np
import API_quippy_thirdorder as FC3
import thirdorder_core
import thirdorder_common
from phonopy import Phonopy
import phonopy.interface.vasp as Intf_vasp
from phonopy.structure.atoms import PhonopyAtoms
import phonopy.file_IO as PhonIO
import API_quippy_phonopy_VASP as api_qpv # remember to set this module to python path
import os, glob
import os.path
import shutil
import sys

# In[2]:

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

    gp_xml_file=sys.argv[5]


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
    print(list4)
    nirred = len(list4)
    nruns = 4 * nirred
    width = len(str(4 * (len(list4) + 1)))
    namepattern = "3RD.POSCAR.{{0:0{0}d}}".format(width)

# generate snapshots to calc FC3 and convert to quippy atom objects.
    
    phipart = np.zeros((3, nirred, ntot))
    p = FC3.build_unpermutation(sposcar)
    for i, e in enumerate(list4):
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
            Scell_quip = api_qpv.phonopyAtoms_to_aseAtoms(Scell)
            force = np.array(api_qpv.calc_force_GAP(gp_xml_file,Scell_quip))
            phipart[:, i, :] -= isign * jsign * force[p, :].T   
     
    phipart =  phipart/(400. * thirdorder_common.H * thirdorder_common.H)
    phifull = thirdorder_core.reconstruct_ifcs(phipart, wedge, list4,poscar, sposcar)
    thirdorder_common.write_ifcs(phifull, poscar, sposcar, dmin, nequi, shifts, frange,"FORCE_CONSTANTS_3RD")
