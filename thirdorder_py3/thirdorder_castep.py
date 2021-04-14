#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from __future__ import print_function
try:
    xrange
except NameError:
    xrange = range

import sys
import os.path
import glob
import shutil
try:
    import cStringIO as StringIO
except ImportError:
    try:
        import StringIO
    except ImportError:
        import io as StringIO

import thirdorder_core
from thirdorder_common import *

# Map element names to atomic numbers
symbol_map = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Uut": 113,
    "Uuq": 114,
    "Uup": 115,
    "Uuh": 116,
    "Uus": 117,
    "Uuo": 118,
}


def read_CASTEP_cell(directory):
    """
    Return all the relevant information contained in a .cell file.
    """
    with dir_context(directory):
        nruter = dict()
        nruter["lattvec"] = np.empty((3, 3))
        nruter["elements"] = []
        f = open(seedname + ".cell", "r")
        castep_cell = f.readlines()
        atoms_list = []
        for index, line in enumerate(castep_cell):
            if '%BLOCK LATTICE_CART' in line.upper():
                for i in xrange(3):
                    new_line = index + i + 1
                    nruter["lattvec"][:, i] = [
                        float(j) for j in castep_cell[new_line].split()
                    ]
            elif '%BLOCK POSITIONS_FRAC' in line.upper():
                index_start = index
            elif '%ENDBLOCK POSITIONS_FRAC' in line.upper():
                index_end = index
        nruter["lattvec"] *= 0.1
        for i in range(index_start + 1, index_end):
            atoms_list.append(castep_cell[i].split())
        atoms_list = list(filter(None, atoms_list))
        atoms_list.sort(key=lambda x: symbol_map[x[0]])
        natoms1 = len(atoms_list)
        nruter["positions"] = np.empty((3, natoms1))
        for i in range(natoms1):
            nruter["positions"][:, i] = [
                float(atoms_list[i][j]) for j in xrange(1, 4)
            ]
            nruter["elements"].append(str(atoms_list[i][0]))
        create_indices = nruter["elements"]
        nruter["elements"] = list(set(nruter["elements"]))
        nruter["elements"].sort(key=lambda x: symbol_map[x])
        nruter["numbers"] = np.array(
            [
                int(create_indices.count(nruter["elements"][i]))
                for i in range(len(nruter["elements"]))
            ],
            dtype=np.intc)
        nruter["types"] = []
        for i in xrange(len(nruter["numbers"])):
            nruter["types"] += [i] * nruter["numbers"][i]
    return nruter


def gen_CASTEP_supercell(CASTEP_cell, na, nb, nc):
    """
    Create a dictionary similar to the first argument but describing a
    supercell.
    """
    nruter = dict()
    nruter["na"] = na
    nruter["nb"] = nb
    nruter["nc"] = nc
    nruter["lattvec"] = np.array(CASTEP_cell["lattvec"])
    nruter["lattvec"][:, 0] *= na
    nruter["lattvec"][:, 1] *= nb
    nruter["lattvec"][:, 2] *= nc
    nruter["elements"] = copy.copy(CASTEP_cell["elements"])
    nruter["numbers"] = na * nb * nc * CASTEP_cell["numbers"]
    nruter["positions"] = np.empty(
        (3, CASTEP_cell["positions"].shape[1] * na * nb * nc))
    pos = 0
    for pos, (k, j, i, iat) in enumerate(
            itertools.product(
                xrange(nc),
                xrange(nb),
                xrange(na), xrange(CASTEP_cell["positions"].shape[1]))):
        nruter["positions"][:, pos] = (
            CASTEP_cell["positions"][:, iat] + [i, j, k]) / [na, nb, nc]
    nruter["types"] = []
    for i in xrange(na * nb * nc):
        nruter["types"].extend(CASTEP_cell["types"])
    return nruter


def write_CASTEP_cell(CASTEP_cell, filename):
    """
    Write the contents of <seedname>.cell to filename.
    """
    f = open(seedname + ".cell", "r")
    castep_cell = f.readlines()
    global hashes
    f = StringIO.StringIO()
    f.write("%BLOCK LATTICE_CART\n")
    for i in xrange(3):
        f.write("{0[0]:>20.15f} {0[1]:>20.15f} {0[2]:>20.15f}\n".format((
            CASTEP_cell["lattvec"][:, i] * 10).tolist()))
    f.write("%ENDBLOCK LATTICE_CART\n")
    f.write("\n")
    f.write("%BLOCK POSITIONS_FRAC\n")
    k = 0
    for i in xrange(len(CASTEP_cell["numbers"])):
        for j in xrange(CASTEP_cell["numbers"][i]):
            l = k + j
            f.write("{0}".format("".join(CASTEP_cell["elements"][i])))
            f.write("{0[0]:>20.15f} {0[1]:>20.15f} {0[2]:>20.15f}\n".format(
                CASTEP_cell["positions"][:, l].tolist()))
        k += j + 1
    f.write("%ENDBLOCK POSITIONS_FRAC\n")

    # Copy everything after '%ENDBLOCK POSITIONS_FRAC'
    for index, line in enumerate(castep_cell):
        if '%ENDBLOCK POSITIONS_FRAC' in line.upper():
            index_end = index
    for i in xrange(index_end + 1, len(castep_cell)):
        f.write(castep_cell[i])
    with open(filename, "w") as finalf:
        finalf.write(f.getvalue())
    f.close()


def normalize_CASTEP_supercell(CASTEP_supercell):
    """
    Rearrange CASTEP_supercell, as generated by gen_CASTEP_supercell, 
    so that it is in valid order, and return the result.
    """
    nruter = copy.deepcopy(CASTEP_supercell)
    # Order used internally (from most to least significant):
    # k,j,i,iat For CASTEP, iat must be the most significant index,
    # i.e., atoms of the same element must go together.
    indices = np.array(xrange(nruter["positions"].shape[1])).reshape(
        (CASTEP_supercell["nc"], CASTEP_supercell["nb"],
         CASTEP_supercell["na"], -1))
    indices = np.rollaxis(indices, 3, 0).flatten().tolist()
    nruter["positions"] = nruter["positions"][:, indices]
    nruter["types"].sort()
    return nruter


def read_forces(filename):
    """
    Read a set of forces on atoms from <seedname>.castep.
    """
    f = open(filename, "r")
    castep_forces = f.readlines()
    f.close()
    nruter = []
    for index, line in enumerate(castep_forces):
        if 'Total number of ions in cell' in line:
            n_atoms = int(line.split()[7])
        if 'Cartesian components (eV/A)' in line:
            starting_line = index + 4
            for i in range(n_atoms):
                f = starting_line + i
                nruter.append(
                    [float(castep_forces[f].split()[m]) for m in range(3, 6)])
    nruter = np.array(nruter, dtype=np.double)
    return nruter


def build_unpermutation(CASTEP_supercell):
    """
    Return a list of integers mapping the atoms in the normalized
    version of CASTEP_supercell to their original indices.
    """
    indices = np.array(xrange(CASTEP_supercell["positions"].shape[1])).reshape(
        (CASTEP_supercell["nc"], CASTEP_supercell["nb"],
         CASTEP_supercell["na"], -1))
    indices = np.rollaxis(indices, 3, 0).flatten()

    return indices.argsort().tolist()


if __name__ == "__main__":
    if len(sys.argv) != 7 or sys.argv[1] not in ("sow", "reap"):
        sys.exit("Usage: {0} sow|reap na nb nc cutoff[nm/-integer] seedname".
                 format(sys.argv[0]))
    action = sys.argv[1]
    na, nb, nc = [int(i) for i in sys.argv[2:5]]
    seedname = sys.argv[6]
    if min(na, nb, nc) < 1:
        sys.exit("Error: na, nb and nc must be positive integers")
    if sys.argv[5][0] == "-":
        try:
            nneigh = -int(sys.argv[5])
        except ValueError:
            sys.exit("Error: invalid cutoff")
        if nneigh == 0:
            sys.exit("Error: invalid cutoff")
    else:
        nneigh = None
        try:
            frange = float(sys.argv[5])
        except ValueError:
            sys.exit("Error: invalid cutoff")
        if frange == 0.:
            sys.exit("Error: invalid cutoff")
    print("Reading %s.cell" % str(seedname))
    CASTEP_cell = read_CASTEP_cell(".")
    natoms = len(CASTEP_cell["types"])
    print("Analyzing the symmetries")
    symops = thirdorder_core.SymmetryOperations(
        CASTEP_cell["lattvec"], CASTEP_cell["types"],
        CASTEP_cell["positions"].T, SYMPREC)
    print("- Symmetry group {0} detected".format(symops.symbol))
    print("- {0} symmetry operations".format(symops.translations.shape[0]))
    print("Creating the supercell")
    CASTEP_supercell = gen_CASTEP_supercell(CASTEP_cell, na, nb, nc)
    ntot = natoms * na * nb * nc
    print("Computing all distances in the supercell")
    dmin, nequi, shifts = calc_dists(CASTEP_supercell)
    if nneigh != None:
        frange = calc_frange(CASTEP_cell, CASTEP_supercell, nneigh, dmin)
        print("- Automatic cutoff: {0} nm".format(frange))
    else:
        print("- User-defined cutoff: {0} nm".format(frange))
    print("Looking for an irreducible set of third-order IFCs")
    wedge = thirdorder_core.Wedge(CASTEP_cell, CASTEP_supercell, symops, dmin,
                                  nequi, shifts, frange)
    print("- {0} triplet equivalence classes found".format(wedge.nlist))
    list4 = wedge.build_list4()
    nirred = len(list4)
    nruns = 4 * nirred
    print("- {0} DFT runs are needed".format(nruns))
    if action == "sow":
        print(sowblock)
        print("Writing undisplaced coordinates to 3RD.%s.cell" % str(seedname))
        write_CASTEP_cell(
            normalize_CASTEP_supercell(CASTEP_supercell),
            "3RD." + str(seedname) + ".cell")
        width = len(str(4 * (len(list4) + 1)))
        namepattern = "3RD." + str(seedname) + ".{{0:0{0}d}}.cell".format(
            width)
        print("Writing displaced coordinates to 3RD.%s.*.cell" % str(seedname))
        for i, e in enumerate(list4):
            for n in xrange(4):
                isign = (-1)**(n // 2)
                jsign = -(-1)**(n % 2)
                # Start numbering the files at 1 for aesthetic
                # reasons.
                number = nirred * n + i + 1
                displaced_CASTEP_supercell = normalize_CASTEP_supercell(
                    move_two_atoms(CASTEP_supercell, e[1], e[3], isign * H,
                                   e[0], e[2], jsign * H))
                filename = namepattern.format(number)
                write_CASTEP_cell(displaced_CASTEP_supercell, filename)
# Copy supercell .cell files to <seedname>-3RD/job-<number>
        cell_names = "3RD.%s.*.cell" % seedname
        indices_list = []
        for i in range(len(sorted(glob.glob(cell_names)))):
            indices_list.append(
                str(i + 1).zfill(len(str(len(sorted(glob.glob(cell_names)))))))
        j = 0
        for i in sorted(glob.glob(cell_names)):
            newpath = ((r'./%s-3RD/job-%s') % (seedname, indices_list[j]))
            j += 1
            if not os.path.exists(newpath): os.makedirs(newpath)
            shutil.move('%s' % i, '%s/%s.cell' % (newpath, seedname))
            # Add .param file to every directory.
            if not glob.glob('%s.param' % seedname):
                sys.exit('%s.param ' % seedname + 'was not found.'
                         ' Please provide a suitable .param file'
                         'for a singlepoint energy calculation and try again!')
            else:
                shutil.copy2('%s.param' % seedname, newpath)
        print("\nCASTEP input files moved to %s-3RD successfully " %
              str(seedname))
        print("{0} CASTEP jobs are prepared for submission".format(nruns))
    else:
        print(reapblock)
        print("Waiting for a list of %s.castep files on stdin" % str(seedname))
        filelist = []
        for l in sys.stdin:
            s = l.strip()
            if len(s) == 0:
                continue
            filelist.append(s)
        nfiles = len(filelist)
        print("- {0} filenames read".format(nfiles))
        if nfiles != nruns:
            sys.exit("Error: {0} filenames were expected".format(nruns))
        for i in filelist:
            if not os.path.isfile(i):
                sys.exit("Error: {0} is not a regular file".format(i))
        print("Reading the forces")
        p = build_unpermutation(CASTEP_supercell)
        forces = []
        for i in filelist:
            forces.append(read_forces(i)[p, :])
            print("- {0} read successfully".format(i))
            res = forces[-1].mean(axis=0)
            print("- \t Average force:")
            print("- \t {0} eV/(A * atom)".format(res))
        print("Computing an irreducible set of anharmonic force constants")
        phipart = np.zeros((3, nirred, ntot))
        for i, e in enumerate(list4):
            for n in xrange(4):
                isign = (-1)**(n // 2)
                jsign = -(-1)**(n % 2)
                number = nirred * n + i
                phipart[:, i, :] -= isign * jsign * forces[number].T
        phipart /= (400. * H * H)
        print("Reconstructing the full array")
        phifull = thirdorder_core.reconstruct_ifcs(
            phipart, wedge, list4, CASTEP_cell, CASTEP_supercell)
        print("Writing the constants to FORCE_CONSTANTS_3RD")
        write_ifcs(phifull, CASTEP_cell, CASTEP_supercell, dmin, nequi, shifts,
                   frange, "FORCE_CONSTANTS_3RD")
    print(doneblock)
