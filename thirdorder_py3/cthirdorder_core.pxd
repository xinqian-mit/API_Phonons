#  thirdorder, help compute anharmonic IFCs from minimal sets of displacements
#  Copyright (C) 2012-2018 Wu Li <wu.li.phys2011@gmail.com>
#  Copyright (C) 2012-2018 Jesús Carrete Montaña <jcarrete@gmail.com>
#  Copyright (C) 2012-2018 Natalio Mingo Bisquert <natalio.mingo@cea.fr>
#  Copyright (C) 2014-2018 Antti J. Karttunen <antti.j.karttunen@iki.fi>
#  Copyright (C) 2016-2018 Genadi Naydenov <gan503@york.ac.uk>
#
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

# This file contains declarations needed by the Cython wrapper around
# spglib.

# Prototype declarations for the small part of spglib wrapped here.
# A single function is enough to get all the symmetry information.
cdef extern from "spglib/spglib.h":
  ctypedef struct SpglibDataset:
    int spacegroup_number
    int hall_number
    char international_symbol[11]
    char hall_symbol[17]
    double transformation_matrix[3][3]
    double origin_shift[3]
    int n_operations
    int (*rotations)[3][3]
    double (*translations)[3]
    int n_atoms
    int *wyckoffs
    int *equivalent_atoms
  SpglibDataset *spg_get_dataset(double lattice[3][3],
                                 double position[][3],
                                 int types[],
                                 int num_atom,
                                 double symprec)
  void spg_free_dataset(SpglibDataset *dataset)
