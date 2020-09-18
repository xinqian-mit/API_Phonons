# Get FC2 and FC3. FC2 will be full mode, FC3 will be compact for ShengBTE
import sys
import numpy as np 
from collections import namedtuple
from itertools import product
from HiPhive_Supercell import Supercell
    
def get_fc2_fc3(phonon,displacements,forces,is_compact_fc=False,options=None,log_level=0):
    supercell = phonon.get_supercell()
    primitive = phonon.get_primitive()
    p2s_map = primitive.p2s_map
    atom_list = phonon.get_primitive().p2s_map
    
    #fc2 = run_alm(supercell,primitive,displacements,forces,1,is_compact_fc=False,options=options,log_level=log_level)
    fcs = run_alm(supercell,primitive,displacements,forces,2,is_compact_fc=is_compact_fc,options=options,log_level=log_level)
    fc2 = fcs[0] # compact fc2
    fc3 = fcs[1]
    #if is_compact_fc:
    #    fc2 = compact_fc_to_full_fc(phonon,fc2) # This line is for converting compact to full FC2
    return fc2,fc3

def run_alm(supercell,
            primitive,
            displacements,
            forces,
            maxorder=2,
            is_compact_fc=False,
            options=None,
            log_level=0):
    fcs = None  # This is returned.

    lattice = supercell.cell
    positions = supercell.scaled_positions
    numbers = supercell.numbers
    natom = len(numbers)
    p2s_map = primitive.p2s_map
    p2p_map = primitive.p2p_map

    alm_options = _update_options(options)
    #print(alm_options)
    num_elems = len(np.unique(numbers))
    if log_level:
        print("---------------------------------"
              " ALM start "
              "--------------------------------")
        print("ALM is a non-trivial force constants calculator. "
              "Please cite the paper:")
        print("T. Tadano and S. Tsuneyuki, "
              "J. Phys. Soc. Jpn. 87, 041015 (2018).")
        print("ALM is developed at https://github.com/ttadano/ALM by T. "
              "Tadano.")
    if log_level == 1:
        print("Increase log-level to watch detailed ALM log.")

    if 'norder' in alm_options:
        _maxorder = alm_options['norder']
    elif 'maxorder' in alm_options:
        _maxorder = alm_options['maxorder']
    else:
        _maxorder = maxorder

    shape = (_maxorder, num_elems, num_elems)
    cutoff_radii = -np.ones(shape, dtype='double')
    if alm_options['cutoff'] is not None:
        if len(alm_options['cutoff']) == 1:
            cutoff_radii[maxorder-1,:,:] = alm_options['cutoff'][0] # only update the cutoff for fc3
        elif np.prod(shape) == len(alm_options['cutoff']):
            cutoff_radii[maxorder-1,:,:] = np.reshape(alm_options['cutoff'], shape)
        else:
            raise RuntimeError("Cutoff is not properly set.")

    _disps, _forces, df_msg = _slice_displacements_and_forces(
        displacements,
        forces,
        alm_options['ndata'],
        alm_options['nstart'],
        alm_options['nend'])

    if log_level > 1:
        print("")
        print("  ndata: %d" % len(displacements))
        for key, val in alm_options.items():
            if val is not None:
                print("  %s: %s" % (key, val))
        print("")
        print(" " + "-" * 67)

    if log_level > 0:
        log_level_alm = log_level - 1
    else:
        log_level_alm = 0

    try:
        from alm import ALM, optimizer_control_data_types
    except ImportError:
        raise ImportError("ALM python module was not found.")

    with ALM(lattice, positions, numbers) as alm:
        if log_level > 0:
            if alm_options['cutoff'] is not None:
                for i in range(_maxorder):
                    if _maxorder > 1:
                        print("fc%d" % (i + 2))
                    print(("cutoff" + " %6s" * num_elems)
                          % tuple(alm.kind_names.values()))
                    for r, kn in zip(cutoff_radii[i], alm.kind_names.values()):
                        print(("   %-3s" + " %6.2f" * num_elems)
                              % ((kn, ) + tuple(r)))
            if df_msg is not None:
                print(df_msg)
        if log_level > 1:
            print("")
        sys.stdout.flush()

        alm.output_filename_prefix = alm_options['output_filename_prefix']
        alm.verbosity = log_level_alm
        
        #print(cutoff_radii)

        alm.define(
            _maxorder,
            cutoff_radii=cutoff_radii,
            nbody=alm_options['nbody'],
            symmetrization_basis=alm_options['symmetrization_basis'])
        alm.displacements = _disps
        alm.forces = _forces

        # Mainly for elastic net (or lasso) regression
        optcontrol = {}
        for key in optimizer_control_data_types:
            if key in alm_options:
                optcontrol[key] = alm_options[key]
        if optcontrol:
            alm.optimizer_control = optcontrol
            if ('cross_validation' in optcontrol and
                optcontrol['cross_validation'] > 0):
                alm.optimize(solver=alm_options['solver'])
                optcontrol['cross_validation'] = 0
                optcontrol['l1_alpha'] = alm.cv_l1_alpha
                alm.optimizer_control = optcontrol

        alm.optimize(solver=alm_options['solver'])

        fcs = _extract_fc_from_alm(alm,
                                   natom,
                                   maxorder,
                                   is_compact_fc,
                                   p2s_map=p2s_map,
                                   p2p_map=p2p_map)

    if log_level:
        print("----------------------------------"
              " ALM end "
              "---------------------------------")

    return fcs


def _update_options(fc_calculator_options):
    """Set ALM options with appropriate data types

    fc_calculator_options : str
        This string should be written such as follows:

            "solver = dense, cutoff = 5"

        This string is parsed as collection of settings that are separated by
        comma ','. Each setting has the format of 'option = value'. The value
        is cast to have its appropriate data type for ALM in this method.

    """

    try:
        from alm import optimizer_control_data_types
    except ImportError:
        raise ImportError("ALM python module was not found.")

    # Default settings.
    alm_options = {'solver': 'dense',
                   'ndata': None,
                   'nstart': None,
                   'nend': None,
                   'nbody': None,
                   'cutoff': None,
                   'symmetrization_basis': 'Lattice',
                   'output_filename_prefix': None}

    if fc_calculator_options is not None:
        alm_option_types = {'cutoff': np.double,
                            'maxorder': int,
                            'norder': int,
                            'ndata': int,
                            'nstart': int,
                            'nend': int,
                            'nbody': np.intc,
                            'output_filename_prefix': str,
                            'solver': str,
                            'symmetrization_basis': str}
        alm_option_types.update(optimizer_control_data_types)
        for option_str in fc_calculator_options.split(","):
            key, val = [x.strip() for x in option_str.split('=')[:2]]
            if key.lower() in alm_option_types:
                if alm_option_types[key.lower()] is np.double:
                    option_value = np.array(
                        [float(x) for x in val.split()], dtype='double')
                elif alm_option_types[key.lower()] is np.intc:
                    option_value = np.array(
                        [int(x) for x in val.split()], dtype='intc')
                else:
                    option_value = alm_option_types[key.lower()](val)
                alm_options[key] = option_value
    return alm_options


def _slice_displacements_and_forces(d, f, ndata, nstart, nend):
    msg = None
    if ndata is not None:
        _d = d[:ndata]
        _f = f[:ndata]
        msg = "Number of displacement supercells: %d" % ndata
    elif nstart is not None and nend is not None:
        _d = d[nstart - 1:nend]
        _f = f[nstart - 1:nend]
        msg = "Supercell index range: %d - %d" % (nstart, nend)
    else:
        return d, f, None

    return (np.array(_d, dtype='double', order='C'),
            np.array(_f, dtype='double', order='C'), msg)


def _extract_fc_from_alm(alm,
                         natom,
                         maxorder,
                         is_compact_fc,
                         p2s_map=None,
                         p2p_map=None):
    # Harmonic: order=1, 3rd: order=2, ...
    fcs = []
    for order in range(1, maxorder + 1):
        fc = None
        p2s_map_alm = alm.getmap_primitive_to_supercell()[0]
        if (is_compact_fc and
            len(p2s_map_alm) == len(p2s_map) and
            (p2s_map_alm == p2s_map).all()):
            fc_shape = (
                (len(p2s_map), ) + (natom, ) * order + (3, ) * (order + 1))
            fc = np.zeros(fc_shape, dtype='double', order='C')
            for fc_elem, indices in zip(
                    *alm.get_fc(order, mode='origin', permutation=1)):
                v = indices // 3
                c = indices % 3
                selection = (p2p_map[v[0]], ) + tuple(v[1:]) + tuple(c)
                fc[selection] = fc_elem

        if fc is None:
            if is_compact_fc:
                atom_list = p2s_map
            else:
                atom_list = np.arange(natom, dtype=int)
            fc_shape = (
                (len(atom_list), ) + (natom, ) * order + (3, ) * (order + 1))
            fc = np.zeros(fc_shape, dtype='double', order='C')
            for fc_elem, indices in zip(
                    *alm.get_fc(order, mode='all', permutation=1)):
                v = indices // 3
                idx = np.where(atom_list == v[0])[0]
                if len(idx) > 0:
                    c = indices % 3
                    selection = (idx[0], ) + tuple(v[1:]) + tuple(c)
                    fc[selection] = fc_elem

        fcs.append(fc)

    return fcs


def compact_fc_to_full_fc(phonon, compact_fc, log_level=0):
    fc = np.zeros((compact_fc.shape[1], compact_fc.shape[1], 3, 3),
                  dtype='double', order='C')
    fc[phonon.primitive.p2s_map] = compact_fc
    distribute_force_constants_by_translations(
        fc, phonon.primitive, phonon.supercell)
    if log_level:
        print("Force constants were expanded to full format.")

    return fc


def distribute_force_constants_by_translations(fc, primitive, supercell):
    """Distribute compact fc data to full fc by pure translations

    For example, the input fc has to be prepared in the following way
    in advance:

    fc = np.zeros((compact_fc.shape[1], compact_fc.shape[1], 3, 3),
                  dtype='double', order='C')
    fc[primitive.p2s_map] = compact_fc

    """
    s2p = primitive.s2p_map
    p2s = primitive.p2s_map
    positions = supercell.scaled_positions
    lattice = supercell.cell.T
    diff = positions - positions[p2s[0]]
    trans = np.array(diff[np.where(s2p == p2s[0])[0]],
                     dtype='double', order='C')
    rotations = np.array([np.eye(3, dtype='intc')] * len(trans),
                         dtype='intc', order='C')
    permutations = primitive.get_atomic_permutations()
    distribute_force_constants(fc, p2s, lattice, rotations, permutations)
    
    
def distribute_force_constants(force_constants,
                               atom_list_done,
                               lattice,  # column vectors
                               rotations,  # scaled (fractional)
                               permutations,
                               atom_list=None):
    map_atoms, map_syms = _get_sym_mappings_from_permutations(
        permutations, atom_list_done)
    rots_cartesian = np.array([similarity_transformation(lattice, r)
                               for r in rotations],
                              dtype='double', order='C')
    if atom_list is None:
        targets = np.arange(force_constants.shape[1], dtype='intc')
    else:
        targets = np.array(atom_list, dtype='intc')
    import phonopy._phonopy as phonoc

    phonoc.distribute_fc2(force_constants,
                          targets,
                          rots_cartesian,
                          permutations,
                          np.array(map_atoms, dtype='intc'),
                          np.array(map_syms, dtype='intc'))
    
    
def _get_sym_mappings_from_permutations(permutations, atom_list_done):

    """This can be thought of as computing 'map_atom_disp' and 'map_sym'
    for all atoms, except done using permutations instead of by
    computing overlaps.

    Input:
        * permutations, shape [num_rot][num_pos]
        * atom_list_done

    Output:
        * map_atoms, shape [num_pos].
        Maps each atom in the full structure to its equivalent atom in
        atom_list_done.  (each entry will be an integer found in
        atom_list_done)

        * map_syms, shape [num_pos].
        For each atom, provides the index of a rotation that maps it
        into atom_list_done.  (there might be more than one such
        rotation, but only one will be returned) (each entry will be
        an integer 0 <= i < num_rot)

    """

    assert permutations.ndim == 2
    num_pos = permutations.shape[1]

    # filled with -1
    map_atoms = np.zeros((num_pos,), dtype='intc') - 1
    map_syms = np.zeros((num_pos,), dtype='intc') - 1

    atom_list_done = set(atom_list_done)
    for atom_todo in range(num_pos):
        for (sym_index, permutation) in enumerate(permutations):
            if permutation[atom_todo] in atom_list_done:
                map_atoms[atom_todo] = permutation[atom_todo]
                map_syms[atom_todo] = sym_index
                break
        else:
            text = ("Input forces are not enough to calculate force constants,"
                    "or something wrong (e.g. crystal structure does not "
                    "match).")
            print(textwrap.fill(text))
            raise ValueError

    assert set(map_atoms) & set(atom_list_done) == set(map_atoms)
    assert -1 not in map_atoms
    assert -1 not in map_syms
    return map_atoms, map_syms


def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))



# Convert FC3 to ShengBTE FC3

def write_shengBTE_fc3(filename, fc3 ,phonon, prim, symprec=1e-5, cutoff=np.inf,
                       fc_tol=1e-8):
    """Writes third-order force constants file in shengBTE format.

    Parameters
    -----------
    filename : str
        input file name
    phonon : Phonopy object
    prim : ase.Atoms
        primitive configuration (must be equivalent to structure used in the
        shengBTE calculation)
    symprec : float
        structural symmetry tolerance
    cutoff : float
        all atoms in cluster must be within this cutoff
    fc_tol : float
        if the absolute value of the largest entry in a force constant is less
        than fc_tol it will not be written
    """

    sheng = _fcs_to_sheng(fc3, phonon, prim, symprec, cutoff, fc_tol)

    raw_sheng = _fancy_to_raw(sheng)

    _write_raw_sheng(raw_sheng, filename)



_ShengEntry = namedtuple('Entry', ['site_0', 'site_1', 'site_2', 'pos_1',
                                   'pos_2', 'fc', 'offset_1', 'offset_2'])


def _fancy_to_raw(sheng):
    """
    Converts force constants namedtuple format defined above (_ShengEntry) to
    format used for writing shengBTE files.
    """
    raw_sheng = []
    for entry in sheng:
        raw_entry = list(entry[:6])
        raw_entry[0] += 1
        raw_entry[1] += 1
        raw_entry[2] += 1
        raw_sheng.append(raw_entry)

    return raw_sheng


def _write_raw_sheng(raw_sheng, filename):
    """ See corresponding read function. """

    with open(filename, 'w') as f:
        f.write('{}\n\n'.format(len(raw_sheng)))

        for index, fc3_row in enumerate(raw_sheng, start=1):
            i, j, k, cell_pos2, cell_pos3, fc3_ijk = fc3_row

            f.write('{:5d}\n'.format(index))

            f.write((3*'{:14.10f} '+'\n').format(*cell_pos2))
            f.write((3*'{:14.10f} '+'\n').format(*cell_pos3))
            f.write((3*'{:5d}'+'\n').format(i, j, k))

            for x, y, z in product(range(3), repeat=3):
                f.write((3*' {:}').format(x+1, y+1, z+1))
                f.write('    {:14.10f}\n'.format(fc3_ijk[x, y, z]))
            f.write('\n')


def _fcs_to_sheng(fc3, phonon, prim,symprec, cutoff, fc_tol):
    """ phonon
    """
    #ell = prim.cell
    #print(cell)
    #ScellMat = phonon.get_supercell_matrix()
    supercell_ph = phonon.get_supercell()
    from API_quippy_phonopy_VASP import phonopyAtoms_to_aseAtoms
    scell = phonopyAtoms_to_aseAtoms(supercell_ph)
    
    supercell = Supercell(scell, prim, symprec)
    assert all(scell.pbc) and all(prim.pbc)
    
    

    n_atoms = len(supercell)

    D = scell.get_all_distances(mic=False, vector=True)
    D_mic = scell.get_all_distances(mic=True, vector=True)
    M = np.eye(n_atoms, dtype=bool)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            M[i, j] = (np.allclose(D[i, j], D_mic[i, j], atol=symprec, rtol=0)
                       and np.linalg.norm(D[i, j]) < cutoff)
            M[j, i] = M[i, j]

    data = {}
    for a0 in supercell:
        #offset_a0 = np.linalg.solve(cell.T, a0.position).round(0).astype(int)
        for a1 in supercell:
            if not M[a0.index, a1.index]:
                continue
            for a2 in supercell:
                if not (M[a0.index, a2.index] and M[a1.index, a2.index]):
                    continue
                #p1 = a1.position
                #offset_a1 = np.linalg.solve(cell.T, a1.position).round(0).astype(int)
                #offset_a2 = np.linalg.solve(cell.T, a2.position).round(0).astype(int)

                offset_1 = (np.subtract(a1.offset, a0.offset))
                offset_2 = (np.subtract(a2.offset, a0.offset))
                #for dim in range(3):
                #    if offset_1[dim]<0:
                #        offset_1[dim] = offset_1[dim]+ScellMat[dim,dim]
                #    if offset_2[dim]<0:
                #        offset_2[dim] = offset_2[dim]+ScellMat[dim,dim]     
                #print(offset_a0)
                #print(offset_1)
                #print(offset_2)

                #sites = (a0.tag, a1.tag, a2.tag)
                sites = (a0.site, a1.site, a2.site)

                key = sites + tuple(offset_1) + tuple(offset_2)
                #print(key)
                
                i = a0.index
                j = a1.index
                k = a2.index

                #ijk = (a0.index, a1.index, a2.index)

                fc = fc3[i,j,k,:,:,:]

                if key in data:
                    assert np.allclose(data[key], fc, atol=fc_tol)
                else:
                    data[key] = fc

    sheng = []
    for k, fc in data.items():
        if np.max(np.abs(fc)) < fc_tol:
            continue
        offset_1 = k[3:6]
        pos_1 = np.dot(offset_1, prim.cell)
        offset_2 = k[6:9]
        pos_2 = np.dot(offset_2, prim.cell)
        entry = _ShengEntry(*k[:3], pos_1, pos_2, fc, offset_1, offset_2)
        sheng.append(entry)

    return sheng