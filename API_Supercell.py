import numpy as np

def site_offset_to_spos(site, offset, basis_spos):
    """ Returns the scaled position of an atom at specified site and offset
    relative to the basis in scaled coordinates"""
    return offset + basis_spos[site]


def spos_to_pos(spos, cell):
    """ Returns the Cartesian coordinate given the scaled coordinate and cell
    metric (cell vectors as rows)"""
    return np.dot(spos, cell)


def pos_to_spos(pos, cell):
    """ Inverse of sps_to_pos"""
    return np.linalg.solve(cell.T, pos)


def spos_to_site_offset(spos, basis_spos, symprec):
    """ Returns the site and offset of the atom at the specified scaled
    coordinate given the scaled positions of the basis atoms"""
    for site, sp in enumerate(basis_spos):
        offset = spos - sp
        rounded_offset = offset.round(0).astype(np.int64)
        # TODO: fix tolerance (symprec should be w.r.t. cart. coord.)
        if np.allclose(rounded_offset, offset, rtol=0, atol=symprec):
            return site, rounded_offset
    raise Exception('spos {} not compatible with basis {} using symprec {}'
                    .format(spos, basis_spos, symprec))


def pos_to_site_offset(pos, cell, basis_spos, symprec):
    """ helper to map pos -> spos -> site/offset"""
    spos = pos_to_spos(pos, cell)
    return spos_to_site_offset(spos, basis_spos, symprec)


def site_offset_to_pos(site, offset, cell, basis_spos):
    """ helper to map site/offset -> spos -> pos"""
    spos = site_offset_to_spos(site, offset, basis_spos)
    return spos_to_pos(spos, cell)


class BaseAtom:
    """ This class represents an atom placed in an infinite crustal"""
    def __init__(self, site, offset):
        assert type(site) is int, type(site)
        assert len(offset) == 3, len(offset)
        assert (all(type(i) is int for i in offset) or
                all(type(i) is np.int64 for i in offset)), type(offset[0])
        self._site = site
        self._offset = np.array(offset)

    @property
    def site(self):
        return self._site

    @property
    def offset(self):
        return self._offset

    def astype(self, dtype):
        """ Useful arguments: list, tuple, np.int64"""
        return dtype((self._site, *self._offset))


class Atom(BaseAtom):
    """ This class represents a crystal atom in a given structure"""
    def __init__(self, *args, **kwargs):
        self._structure = kwargs.pop('structure', None)
        super().__init__(*args, **kwargs)

    @property
    def pos(self):
        return site_offset_to_pos(self._site, self._offset,
                                  self._structure.cell,
                                  self._structure.spos)

    @property
    def number(self):
        return self._structure.numbers[self._site]


class SupercellAtom(Atom):
    """ Represents an atom in a supercell but site and offset given by an
    underlying primitve cell"""
    def __init__(self, *args, **kwargs):
        self._index = kwargs.pop('index')
        assert type(self._index) is int
        super().__init__(*args, **kwargs)

    @property
    def index(self):
        return self._index


class Structure:
    """ This class essentially wraps the ase.Atoms class but is a bit more
    carefull about pbc and scaled coordinates. It also returns hiphive.Atom
    objects instead"""
    def __init__(self, atoms, symprec=1e-6):
        spos = atoms.get_scaled_positions(wrap=False)
        for sp in spos.flat:
            if not (-symprec < sp < (1 - symprec)):
                raise ValueError('bad spos {}'.format(sp))
        self._spos = spos
        self._cell = atoms.cell
        self._numbers = atoms.numbers

    def __len__(self):
        return len(self._spos)

    @property
    def spos(self):
        return self._spos

    @property
    def cell(self):
        return self._cell

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('Structure contains {} atoms'.format(len(self)))
        return Atom(index, (0, 0, 0), structure=self)

    def atom_from_pos(self, pos, symprec=None):
        if symprec is None:
            symprec = self._symprec
        site, offset = pos_to_site_offset(pos, self._cell, self._spos, symprec)
        return Atom(site, offset, structure=self)


class Supercell:
    """ This class tries to represent atoms in a supercell as positioned on the
    primitve lattice"""
    def __init__(self, supercell, prim, symprec):
        self._supercell = Structure(supercell)
        self._prim = Structure(prim)
        self._symprec = symprec
        self._map = list()
        self._inverse_map_lookup = dict()
        self._create_map()

    def _create_map(self):
        for atom in self._supercell:
            atom = self._prim.atom_from_pos(atom.pos, self._symprec)
            self._map.append(atom.astype(tuple))

    def wrap_atom(self, atom):
        atom = Atom(atom.site, atom.offset, structure=self._prim)
        tup = atom.astype(tuple)
        index = self._inverse_map_lookup.get(tup, None)
        if index is None:
            atom = self._supercell.atom_from_pos(atom.pos, self._symprec)
            index = atom.site
            self._inverse_map_lookup[tup] = index
        return self[index]

    def index(self, site, offset):
        atom = self.wrap_atom(BaseAtom(site, offset))
        return atom.index

    def __getitem__(self, index):
        tup = self._map[index]
        return SupercellAtom(tup[0], tup[1:], structure=self._prim,
                             index=index)

    def __len__(self):
        return len(self._supercell)

