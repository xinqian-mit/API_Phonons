#!/usr/bin/env python3
# coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
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

gp_xml_file='gp_bcc.xml'
fmax = 1e-6
pot = quippy.potential.Potential(param_filename=gp_xml_file)


ucell = Intf_vasp.read_vasp('POSCAR')
ucell_ase = api_ph.phonopyAtoms_to_aseAtoms(ucell)
ucell_ase.set_calculator(pot) # ase can accept quippy potential object as calculators
ucell_ase.set_constraint(FixAtoms(mask=[True for atom in ucell_ase]))
ucf = UnitCellFilter(ucell_ase)
relax = BFGS(ucf)

relax.run(fmax=fmax)
ucell_ph = api_ph.aseAtoms_to_phonopyAtoms(ucell_ase)
Intf_vasp.write_vasp('POSCAR_relx',ucell_ph)


# In[ ]:




