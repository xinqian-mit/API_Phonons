#!/usr/bin/env python3

from ase.spacegroup import crystal
from ase.visualize import view
from ase.io import write
import ase.build
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
import API_quippy_phonopy_VASP as api_qpv
import API_phonopy_lammps as api_plmp
from phonopy import Phonopy
import phonopy.file_IO as PhonIO
from phonopy.interface.calculator import get_default_physical_units
import copy as cp
import multiprocessing as mp
import phonopy.interface.vasp as phonVasp


# Specify the slab cut and layers
Relax = True # Wether relax file
fmax = 1e-6

a = 5.63095072418881 #5.65414946767 # lattice constant of NaCl
nacl = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
               cellpar=[a, a, a, 90, 90, 90])

Nrepeat = [4,4,4]
ph = Phonopy(nacl,np.diag(Nrepeat),primitive_matrix=np.eye(3)) 
Cell = api_qpv.phonopyAtoms_to_aseAtoms(ph.get_supercell()) # I always use phonopy's supcercell module. 

lammps_header=['units metal',
                   'atom_style charge',
                   'atom_modify map array sort 0 0']

logfile='log.lammps'

cmds = ["pair_style eim","pair_coeff * * Na Cl ffield.eim Na Cl"]

# --------------------------- Relax the Cell  --------------------------- #
if Relax:
   #from ase.calculators.lammpsrun import LAMMPS
   from ase.calculators.lammpslib import LAMMPSlib
   from ase.constraints import FixAtoms,UnitCellFilter
   from ase.optimize import BFGS

   lmps = LAMMPSlib(lmpcmds=cmds, log_file=logfile,lammps_header=lammps_header,keep_alive=True) #LAMMPS(parameters=parameters, files=pot_file)
   Cell.set_calculator(lmps)
   Cell.set_constraint(FixAtoms(mask=[True for atom in Cell]))
   ucf = UnitCellFilter(Cell)
   relax = BFGS(ucf)
   relax.run(fmax=fmax)
   
Scell_dims_str = str(Nrepeat[0])+str(Nrepeat[1])+str(Nrepeat[2])

Struct_prefix = 'NaCl_'+Scell_dims_str
api_plmp.write_R0(Struct_prefix,nacl,ph.get_supercell())
ase.io.write(Struct_prefix+'.cif',Cell,'cif')
Cell.set_pbc([True,True,True])
lmp_Data_filename = Struct_prefix+'.data'
api_plmp.write_lmp_data(lmp_Data_filename,Cell)

phonVasp.write_vasp(Struct_prefix+'.POSCAR',Cell)



