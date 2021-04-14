#!/usr/bin/env python
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
import os
import os.path
import copy
import itertools
import contextlib
try:
    import cStringIO as StringIO
except ImportError:
    try:
        import StringIO
    except ImportError:
        import io as StringIO
try:
    import hashlib
    hashes = True
except ImportError:
    hashes = False
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.spatial
import scipy.spatial.distance

H = 1e-3  # Magnitude of the finite displacements, in nm.
SYMPREC = 1e-5  # Tolerance for symmetry search

sowblock = """
.d88888b   .88888.  dP   dP   dP
88.    "' d8'   `8b 88   88   88
`Y88888b. 88     88 88  .8P  .8P
      `8b 88     88 88  d8'  d8'
d8'   .8P Y8.   .8P 88.d8P8.d8P
 Y88888P   `8888P'  8888' Y88'
ooooooooooooooooooooooooooooooooo
"""
reapblock = """
 888888ba   88888888b  .d888888   888888ba
 88    `8b  88        d8'    88   88    `8b
a88aaaa8P' a88aaaa    88aaaaa88a a88aaaa8P'
 88   `8b.  88        88     88   88
 88     88  88        88     88   88
 dP     dP  88888888P 88     88   dP
oooooooooooooooooooooooooooooooooooooooooooo
"""
doneblock = """
888888ba   .88888.  888888ba   88888888b
88    `8b d8'   `8b 88    `8b  88
88     88 88     88 88     88 a88aaaa
88     88 88     88 88     88  88
88    .8P Y8.   .8P 88     88  88
8888888P   `8888P'  dP     dP  88888888P
ooooooooooooooooooooooooooooooooooooooooo
"""


@contextlib.contextmanager
def dir_context(directory):
    """
    Context manager used to run code in another directory.
    """
    curdir = os.getcwd()
    os.chdir(directory)
    try:
        yield directory
    finally:
        os.chdir(curdir)


def gen_SPOSCAR(poscar, na, nb, nc):
    """
    Create a dictionary similar to the first argument but describing a
    supercell.
    """
    nruter = dict()
    nruter["na"] = na
    nruter["nb"] = nb
    nruter["nc"] = nc
    nruter["lattvec"] = np.array(poscar["lattvec"])
    nruter["lattvec"][:, 0] *= na
    nruter["lattvec"][:, 1] *= nb
    nruter["lattvec"][:, 2] *= nc
    nruter["elements"] = copy.copy(poscar["elements"])
    nruter["numbers"] = na * nb * nc * poscar["numbers"]
    nruter["positions"] = np.empty(
        (3, poscar["positions"].shape[1] * na * nb * nc))
    pos = 0
    for pos, (k, j, i, iat) in enumerate(
            itertools.product(
                xrange(nc),
                xrange(nb), xrange(na), xrange(poscar["positions"].shape[1]))):
        nruter["positions"][:, pos] = (
            poscar["positions"][:, iat] + [i, j, k]) / [na, nb, nc]
    nruter["types"] = []
    for i in xrange(na * nb * nc):
        nruter["types"].extend(poscar["types"])
    return nruter


def calc_dists(sposcar):
    """
    Return the distances between atoms in the supercells, their
    degeneracies and the associated supercell vectors.
    """
    ntot = sposcar["positions"].shape[1]
    posi = np.dot(sposcar["lattvec"], sposcar["positions"])
    d2s = np.empty((27, ntot, ntot))
    for j, (ja, jb, jc) in enumerate(
            itertools.product(xrange(-1, 2), xrange(-1, 2), xrange(-1, 2))):
        posj = np.dot(sposcar["lattvec"],
                      (sposcar["positions"].T + [ja, jb, jc]).T)
        d2s[j, :, :] = scipy.spatial.distance.cdist(posi.T, posj.T,
                                                    "sqeuclidean")
    d2min = d2s.min(axis=0)
    dmin = np.sqrt(d2min)
    degenerate = (np.abs(d2s - d2min) < 1e-4)
    nequi = degenerate.sum(axis=0, dtype=np.intc)
    maxequi = nequi.max()
    shifts = np.empty((ntot, ntot, maxequi))
    sorting = np.argsort(np.logical_not(degenerate), axis=0)
    shifts = np.transpose(sorting[:maxequi, :, :], (1, 2, 0)).astype(np.intc)
    return (dmin, nequi, shifts)


def calc_frange(poscar, sposcar, n, dmin):
    """
    Return the maximum distance between n-th neighbors in the structure.
    """
    natoms = len(poscar["types"])
    tonth = []
    warned = False
    for i in xrange(natoms):
        ds = dmin[i, :].tolist()
        ds.sort()
        u = []
        for j in ds:
            for k in u:
                if np.allclose(k, j):
                    break
            else:
                u.append(j)
        try:
            tonth.append(.5 * (u[n] + u[n + 1]))
        except IndexError:
            if not warned:
                sys.stderr.write(
                    "Warning: supercell too small to find n-th neighbours\n")
                warned = True
            tonth.append(1.1 * max(u))
    return max(tonth)


def move_two_atoms(poscar, iat, icoord, ih, jat, jcoord, jh):
    """
    Return a copy of poscar with atom iat displaced by ih nm along
    its icoord-th Cartesian coordinate and atom jat displaced by
    jh nm along its jcoord-th Cartesian coordinate.
    """
    nruter = copy.deepcopy(poscar)
    disp = np.zeros(3)
    disp[icoord] = ih
    nruter["positions"][:, iat] += scipy.linalg.solve(nruter["lattvec"], disp)
    disp[:] = 0.
    disp[jcoord] = jh
    nruter["positions"][:, jat] += scipy.linalg.solve(nruter["lattvec"], disp)
    return nruter


def write_ifcs(phifull, poscar, sposcar, dmin, nequi, shifts, frange,
               filename):
    """
    Write out the full anharmonic interatomic force constant matrix,
    taking the force cutoff into account.
    """
    natoms = len(poscar["types"])
    ntot = len(sposcar["types"])

    shifts27 = list(
        itertools.product(xrange(-1, 2), xrange(-1, 2), xrange(-1, 2)))
    frange2 = frange * frange

    nblocks = 0
    f = StringIO.StringIO()
    for ii, jj in itertools.product(xrange(natoms), xrange(ntot)):
        if dmin[ii, jj] >= frange:
            continue
        jatom = jj % natoms
        shiftsij = [shifts27[i] for i in shifts[ii, jj, :nequi[ii, jj]]]
        for kk in xrange(ntot):
            if dmin[ii, kk] >= frange:
                continue
            katom = kk % natoms
            shiftsik = [shifts27[i] for i in shifts[ii, kk, :nequi[ii, kk]]]
            d2min = np.inf
            for shift2 in shiftsij:
                carj = np.dot(sposcar["lattvec"],
                              shift2 + sposcar["positions"][:, jj])
                for shift3 in shiftsik:
                    cark = np.dot(sposcar["lattvec"],
                                  shift3 + sposcar["positions"][:, kk])
                    d2 = ((carj - cark)**2).sum()
                    if d2 < d2min:
                        best2 = shift2
                        best3 = shift3
                        d2min = d2
            if d2min >= frange2:
                continue
            nblocks += 1
            Rj = np.dot(sposcar["lattvec"], best2 + sposcar["positions"][:, jj]
                        - sposcar["positions"][:, jatom])
            Rk = np.dot(sposcar["lattvec"], best3 + sposcar["positions"][:, kk]
                        - sposcar["positions"][:, katom])
            f.write("\n")
            f.write("{:>5}\n".format(nblocks))
            f.write("{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".format(
                list(10. * Rj)))
            f.write("{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".format(
                list(10. * Rk)))
            f.write(
                "{:>6d} {:>6d} {:>6d}\n".format(ii + 1, jatom + 1, katom + 1))
            for ll, mm, nn in itertools.product(
                    xrange(3), xrange(3), xrange(3)):
                f.write("{:>2d} {:>2d} {:>2d} {:>20.10e}\n".format(
                    ll + 1, mm + 1, nn + 1, phifull[ll, mm, nn, ii, jj, kk]))
    ffinal = open(filename, "w")
    ffinal.write("{:>5}\n".format(nblocks))
    ffinal.write(f.getvalue())
    f.close()
    ffinal.close()
