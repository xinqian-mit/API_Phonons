#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  Fourthorder, help compute anharmonic IFCs from minimal sets of displacements
#  Copyright (C) 2021 Zherui Han <zrhan@purdue.edu>
#  Copyright (C) 2021 Xiaolong Yang <xiaolongyang1990@gmail.com>
#  Copyright (C) 2021 Wu Li <wu.li.phys2011@gmail.com>
#  Copyright (C) 2021 Tianli Feng <Tianli.Feng2011@gmail.com>
#  Copyright (C) 2021 Xiulin Ruan <ruan@purdue.edu>

#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os.path
import glob
import Fourthorder_core
from Fourthorder_common import *
import phonopy.interface.vasp as Intf_vasp
import os
from ase.calculators.lammpslib import LAMMPSlib
import API_phonopy as api_ph 
import API_phonopy_lammps as api_pl
import API_thirdorder as FC3


if __name__=="__main__":
    
    #Nprocesses = 2
    Nrepeat = [3,3,3]
    nneigh = 2 # neighbor cutoff
    
    cmds = ["pair_style eim","pair_coeff * * Na Cl ffield.eim Na Cl"]
    lammps_header=['units metal',
                   'atom_style atomic',
                   'atom_modify map array sort 0 0']
    
    
    
    na,nb,nc = Nrepeat


    if min(na,nb,nc)<1:
        sys.exit("Error: na, nb and nc must be positive integers")

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
    frange=calc_frange(poscar,sposcar,nneigh,dmin)


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
    
    
    forces = []
    
    p=FC3.build_unpermutation(sposcar)
    phipart=np.zeros((3,nirred,ntot))
    
    for i,e in enumerate(list6):
        for n in range(8):
            isign=(-1)**(n//4)
            jsign=(-1)**(n%4//2)
            ksign=(-1)**(n%2)
            # print e[2],e[5],isign,e[1],e[4],jsign,e[0],e[3],ksign
            # Start numbering the files at 1 for aesthetic
            # reasons.
            number=nirred*n+i+1
            dsposcar=FC3.normalize_SPOSCAR(
                move_three_atoms(sposcar,
                               e[2],e[5],isign*H,
                               e[1],e[4],jsign*H,
                               e[0],e[3],ksign*H))
            filename=namepattern.format(number)
            FC3.write_POSCAR(dsposcar,filename)
            
            Scell = Intf_vasp.read_vasp(filename)
            scell = api_ph.phonopyAtoms_to_aseAtoms(Scell)
            os.remove(filename)
            
            lmp = LAMMPSlib(lmpcmds=cmds, log_file='log.'+str(i)+'.'+str(n),lammps_header=lammps_header)
            scell.calc = lmp
            force = scell.get_forces()
            forces.append(force[p,:])
            
            phipart[:,i,:] -= isign*jsign*ksign*force[p,:].T
            
            os.remove('log.'+str(i)+'.'+str(n))

    res=forces[-1].mean(axis=0)
    print("- \t Average force:")
    print("- \t {0} eV/(A * atom)".format(res))
    print("Computing an irreducible set of anharmonic force constants")

    phipart/=(8000.*H*H*H)
    print("Reconstructing the full array")
    phifull=Fourthorder_core.reconstruct_ifcs(phipart,wedge,list6,poscar,sposcar)
    print("Writing the constants to FORCE_CONSTANTS_4TH")
    write_ifcs(phifull,poscar,sposcar,dmin,nequi,shifts,frange,"FORCE_CONSTANTS_4TH")


