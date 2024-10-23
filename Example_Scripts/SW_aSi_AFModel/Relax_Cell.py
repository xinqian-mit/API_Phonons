

import numpy as np
from phonopy import Phonopy
import phonopy.interface.vasp as Intf_vasp
from phonopy.structure.atoms import PhonopyAtoms
import phonopy.file_IO as PhonIO
from phonopy.interface.calculator import get_default_physical_units
import API_phonopy as api_ph # remember to set this module to python path
from ase.optimize import BFGS
import API_phonopy_lammps as api_pl
from ase.calculators.lammpslib import LAMMPSlib
from ase.constraints import UnitCellFilter
from ase.constraints import FixAtoms
import quippy


fmax = 1e-6

in_data_file = 'POSCAR_512'
relx_data_file = 'POSCAR_512_SWrelx'

pot_file ='Si.sw'
lammps_header = [
    'units metal',
    'atom_style atomic',
    'atom_modify map array sort 0 0',
    #'box tilt large',
]

cmds = [
    "pair_style sw",
    "pair_coeff * * Si.sw Si",
]

lmp = LAMMPSlib(lmpcmds=cmds, log_file='log.lammps',lammps_header=lammps_header,keep_alive=True)

ucell = Intf_vasp.read_vasp(in_data_file)
ucell_ase = api_ph.phonopyAtoms_to_aseAtoms(ucell)
ucell_ase.set_calculator(lmp) # ase can accept quippy potential object as calculators
#ucell_ase.set_constraint(FixAtoms(mask=[True for atom in ucell_ase]))
#ucf = UnitCellFilter(ucell_ase)
relax = BFGS(ucell_ase)

relax.run(fmax=fmax)
ucell_ph = api_ph.aseAtoms_to_phonopyAtoms(ucell_ase)
Intf_vasp.write_vasp(relx_data_file,ucell_ph)


# In[ ]:




