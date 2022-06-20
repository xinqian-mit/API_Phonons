from ase.calculators.lammpslib import LAMMPSlib
import numpy as np

import ase.io as io
from matscipy.neighbours import neighbour_list


import API_phonopy as api_ph

from numba import njit 

## ------------------------------------- Get Lammps box parameters -----------------------------------------------------##



def get_lmp_boxbounds(Scell):
    vec_lo = Scell.get_celldisp()
    xlo = vec_lo[0][0]; ylo = vec_lo[1][0]; zlo = vec_lo[2][0];

    [A,B,C] = Scell.get_cell()
    Ah = A/np.linalg.norm(A)
    Bh = B/np.linalg.norm(B)
    Ch = C/np.linalg.norm(C)
    AxB = np.cross(A,B)
    AxB_h = AxB/np.linalg.norm(AxB)

    ax = np.linalg.norm(A)
    bx = np.dot(B,Ah)
    by = np.linalg.norm(np.cross(Ah,B))
    cx = np.dot(C,Ah)
    cy = np.dot(B,C)-bx*cx/by
    cz = np.sqrt(np.dot(C,C)-cx*cx-cy*cy)

    [a,b,c,alpha,beta,gamma]=Scell.get_cell_lengths_and_angles()
    cos_alpha = np.cos(alpha/180*np.pi)
    cos_beta = np.cos(alpha/180*np.pi)
    cos_gamma = np.cos(alpha/180*np.pi)

    lx = a 
    xy = b*cos_gamma
    xz = c*cos_beta
    ly = np.sqrt(b*b-xy*xy)
    yz = (b*c*cos_alpha-xy*xz)/ly
    lz = np.sqrt(c*c-xz*xz-yz*yz)

    xhi = xlo + lx
    yhi = ylo + ly
    zhi = zlo + lz

    xlo_bound = xlo + np.min([0.0,xy,xz,xy+xz])
    xhi_bound = xhi + np.max([0.0,xy,xz,xy+xz])
    ylo_bound = ylo + np.min([0.0,yz])
    yhi_bound = yhi + np.min([0.0,yz])
    zlo_bound = zlo
    zhi_bound = zhi
    return xlo_bound,xhi_bound,ylo_bound,yhi_bound,zlo_bound,zhi_bound,xy,xz,yz



def calc_lmp_force_sets(cmds,Scells_ph,atomtypes='atomic',logfile='log.lammps',lammps_header=[],
                        create_atoms=True, create_box=True, boundary=True, keep_alive=False): 
    """
    This function uses ase and lammps' python API to calculate forces. Comment this funciton if it's not installed.
    In cmd, specifies the potential    
    Scells takes the list of perturbated supercells,  phonopyatom objs.
    
    """
    if lammps_header == []:
        lammps_header=['units metal',
                       'atom_style '+atomtypes,
                       'atom_modify map array sort 0 0']        
        
    

    if type(Scells_ph)!=list:
        Scells_ph = [Scells_ph]
    
    force_scells=[]
    for scell_ph in Scells_ph:
        lammps = LAMMPSlib(lmpcmds=cmds, log_file=logfile,lammps_header=lammps_header,
                           create_atoms=create_atoms, create_box=create_box, boundary=boundary, keep_alive=keep_alive) # lammps obj has to be in the loop.
        scell = api_ph.phonopyAtoms_to_aseAtoms(scell_ph)
        scell.set_calculator(lammps)
        forces = scell.get_forces()
        force_scells.append(forces.tolist())
    
    return force_scells


def calc_lmp_force(cmds,Scell_ph,atomtypes='atomic',logfile='log.lammps',lammps_header=[],
                   create_atoms=True, create_box=True, boundary=True, keep_alive=False): 
    """
    This function uses ase and lammps' python API to calculate forces. 
    In cmd, specifies the potential    
    Scells takes the list of perturbated supercells,  phonopyatom objs.
    
    Settings in the lammps header will overwrite the settings of the atmotypes, if lammps_header is
    explictly specified. 
    
    """
    if lammps_header == []:
        lammps_header=['units metal',
                       'atom_style '+atomtypes,
                       'atom_modify map array sort 0 0']  



    lammps = LAMMPSlib(lmpcmds=cmds, log_file=logfile,lammps_header=lammps_header,
                       create_atoms=create_atoms, create_box=create_box, boundary=boundary, keep_alive=keep_alive) # lammps obj has to be in the loop.
    scell = api_ph.phonopyAtoms_to_aseAtoms(Scell_ph)
    scell.set_calculator(lammps)
    forces = scell.get_forces()
    
    return forces


def get_DFSETS_lmp(Scell0,Scell_snaps,cmds,atomtypes='atomic',logfile='log.lammps',lammps_header=[],
                   create_atoms=True, create_box=True, boundary=True, keep_alive=False): 
    # Scell0 & Scell_snaps are phonopy atom objects. Scell0 is the unperturbated supercell,
    # Scell_snaps are perturbated ones.
    Nsnaps = len(Scell_snaps)
    pos0_frac = Scell0.get_scaled_positions()
    latt_vec = Scell0.get_cell()
    displacements = np.zeros([Nsnaps,Scell0.get_number_of_atoms(),3])
    forces = np.zeros([Nsnaps,Scell0.get_number_of_atoms(),3])
    for i,scell in  enumerate(Scell_snaps):
        #print(i)
        pos_frac = scell.get_scaled_positions()
        ur = pos_frac-pos0_frac
        ui = np.zeros(pos_frac.shape)
        fi = np.zeros(pos_frac.shape)
        for iat in range(scell.get_number_of_atoms()):
            for j in range(3): #periodic boundary condition, wrap the sscell vec
                ujr = ur[iat][j]
                if (np.abs(ujr)>np.abs(ujr+1)):
                    ur[iat][j] = ujr+1
                if (np.abs(ujr)>np.abs(ujr-1)):
                    ur[iat][j] = ujr-1 
            ui[iat][0]=ur[iat][0]*latt_vec[0][0]+ur[iat][1]*latt_vec[1][0]+ur[iat][2]*latt_vec[2][0] #get disps
            ui[iat][1]=ur[iat][0]*latt_vec[0][1]+ur[iat][1]*latt_vec[1][1]+ur[iat][2]*latt_vec[2][1]
            ui[iat][2]=ur[iat][0]*latt_vec[0][2]+ur[iat][1]*latt_vec[1][2]+ur[iat][2]*latt_vec[2][2]
        scell_ase = api_ph.phonopyAtoms_to_aseAtoms(scell)
        fi = calc_lmp_force(cmds,scell_ase,atomtypes,logfile,lammps_header=lammps_header,
                            create_atoms=create_atoms, create_box=create_box, boundary=boundary, keep_alive=keep_alive) # get forces
        displacements[i][:][:]=ui
        forces[i][:][:]=fi
    
    return displacements,forces


#---------------------------------------------File io--------------------------------------------------#
def read_extxyz(filename,index=':'):
    Snaps = io.read(filename, format='extxyz',index=index)
    energies = []
    forces = []
    for snap in Snaps:
        energies.append(snap.get_calculator().results['energy'])
        forces.append(snap.arrays['force'])
        
    energies = np.array(energies)
    forces = np.array(forces)
    return Snaps, energies,forces


def write_ScellCar_MaterStudio(Prefix,ucell,Nrepeat,Element_atypes,Symbol_atypes):
    Na = Nrepeat[0]
    Nb = Nrepeat[1]
    Nc = Nrepeat[2]
    fid = open(Prefix+str(Na)+str(Nb)+str(Nc)+'.car','w')
    
    Nbasis = ucell.get_global_number_of_atoms()
    atyp_ucell = ucell.get_tags() # remember to set the type as tags
    ucell_vec = ucell.get_cell()
    mol_id = ucell.get_array('mol-id')
    pos_ucell = ucell.get_positions()
    charges_ucell = ucell.get_initial_charges()
    #Natoms = Na*Nb*Nc*Nbasis
    
    Elements = np.unique(Element_atypes)
                        
    Num_ele = np.zeros(len(Elements),dtype='int64')
    
    alpha = np.arccos(np.dot(ucell_vec[1],ucell_vec[2])/np.linalg.norm(ucell_vec[1])/np.linalg.norm(ucell_vec[2]))/np.pi*180
    beta = np.arccos(np.dot(ucell_vec[2],ucell_vec[0])/np.linalg.norm(ucell_vec[2])/np.linalg.norm(ucell_vec[0]))/np.pi*180
    gamma = np.arccos(np.dot(ucell_vec[0],ucell_vec[1])/np.linalg.norm(ucell_vec[0])/np.linalg.norm(ucell_vec[1]))/np.pi*180
    
    La = np.linalg.norm(ucell_vec[0])*Na
    Lb = np.linalg.norm(ucell_vec[1])*Nb
    Lc = np.linalg.norm(ucell_vec[2])*Nc
    
    Nmols_ucell = len(np.unique(mol_id))
    
    mol_id_max = Nmols_ucell*Na*Nb*Nc
    len_id = len(str(mol_id_max))
    
    
    
    fid.write('!BIOSYM archive 3\n');
    fid.write('PBC=ON\n');
    fid.write('Materials Studio Generated CAR File\n');
    fid.write('!DATE Sat Mar 12 15:36:48 2016\n');
    fid.write('PBC    {:.4f}    {:.4f}   {:.4f}   {:.4f}   {:.4f}  {:.4f} (P1)\n'.format(La,Lb,Lc,alpha,beta,gamma));
       
    
    Num_ele_ucell = np.zeros(len(Elements),dtype='int64')
    for ib in range(Nbasis):
        atype = atyp_ucell[ib]-1
        ele_atype_ib = Element_atypes[atype]
        for iele,sym_ele in enumerate(Elements):
            if sym_ele == ele_atype_ib:
                Num_ele_ucell[iele] += 1
                
    #max_num_ele = len(str(np.max(Num_ele_ucell*Na*Nb*Nc)))+2
    fom='{:4s}    {:-13.9f}  {:-13.9f}  {:-13.9f} XXXX {:' + str(len_id)+ 'g}      {:3s}      {:2s}  {:-.4f}\n'; 
        
    

    
    for ib in range(Nbasis):
        icell = 0
        for iz in range(Nc):
            for iy in range(Nb):
                for ix in range(Na):
                    
                    pos = pos_ucell[ib,:] + ix*ucell_vec[0,:] + iy*ucell_vec[1,:] + iz*ucell_vec[2,:]
                    atype = atyp_ucell[ib]-1 # I'm assuming the tags start counting from 1.
                    element = Element_atypes[atype]
                    symbol = Symbol_atypes[atype]
                    
                    for iele,sym_ele in enumerate(Elements):
                        if sym_ele == element:
                            Num_ele[iele] +=1
                            aaint = Num_ele[iele]%Num_ele_ucell[iele]
                            if aaint == 0:
                                aaint = Num_ele_ucell[iele]
                            aindex = element+str(aaint)

                    fid.write(fom.format(aindex,pos[0],pos[1],pos[2],mol_id[ib]+icell*Nmols_ucell,symbol,element,charges_ucell[ib]))
                    icell = icell +1
    fid.write('end')
    fid.close()                                      
                    

def write_ScellCar_MaterStudio_ucell(Prefix,ucell,Nrepeat,Element_atypes,Symbol_atypes):
    # This function has different for loops, where the ib is inner most loop.
    Na = Nrepeat[0]
    Nb = Nrepeat[1]
    Nc = Nrepeat[2]
    fid = open(Prefix+str(Na)+str(Nb)+str(Nc)+'.car','w')
    
    Nbasis = ucell.get_global_number_of_atoms()
    atyp_ucell = ucell.get_tags() # remember to set the type as tags
    ucell_vec = ucell.get_cell()
    mol_id = ucell.get_array('mol-id')
    pos_ucell = ucell.get_positions()
    charges_ucell = ucell.get_initial_charges()
    #Natoms = Na*Nb*Nc*Nbasis
    
    Elements = np.unique(Element_atypes)
                        
    Num_ele = np.zeros(len(Elements),dtype='int64')
    
    alpha = np.arccos(np.dot(ucell_vec[1],ucell_vec[2])/np.linalg.norm(ucell_vec[1])/np.linalg.norm(ucell_vec[2]))/np.pi*180
    beta = np.arccos(np.dot(ucell_vec[2],ucell_vec[0])/np.linalg.norm(ucell_vec[2])/np.linalg.norm(ucell_vec[0]))/np.pi*180
    gamma = np.arccos(np.dot(ucell_vec[0],ucell_vec[1])/np.linalg.norm(ucell_vec[0])/np.linalg.norm(ucell_vec[1]))/np.pi*180
    
    La = np.linalg.norm(ucell_vec[0])*Na
    Lb = np.linalg.norm(ucell_vec[1])*Nb
    Lc = np.linalg.norm(ucell_vec[2])*Nc
    
    Nmols_ucell = len(np.unique(mol_id))
    
    mol_id_max = Nmols_ucell*Na*Nb*Nc
    len_id = len(str(mol_id_max))
    
    
    
    fid.write('!BIOSYM archive 3\n');
    fid.write('PBC=ON\n');
    fid.write('Materials Studio Generated CAR File\n');
    fid.write('!DATE Sat Mar 12 15:36:48 2016\n');
    fid.write('PBC    {:.4f}    {:.4f}   {:.4f}   {:.4f}   {:.4f}  {:.4f} (P1)\n'.format(La,Lb,Lc,alpha,beta,gamma));
       
    
    Num_ele_ucell = np.zeros(len(Elements),dtype='int64')
    for ib in range(Nbasis):
        atype = atyp_ucell[ib]-1
        ele_atype_ib = Element_atypes[atype]
        for iele,sym_ele in enumerate(Elements):
            if sym_ele == ele_atype_ib:
                Num_ele_ucell[iele] += 1
                
    #max_num_ele = len(str(np.max(Num_ele_ucell*Na*Nb*Nc)))+2
    fom='{:4s}    {:-13.9f}  {:-13.9f}  {:-13.9f} XXXX {:' + str(len_id)+ 'g}      {:3s}      {:2s}  {:-.4f}\n'; 
        
    

    
    
    icell = 0
    for iz in range(Nc):
        for iy in range(Nb):
            for ix in range(Na):
                for ib in range(Nbasis):
                    
                    pos = pos_ucell[ib,:] + ix*ucell_vec[0,:] + iy*ucell_vec[1,:] + iz*ucell_vec[2,:]
                    atype = atyp_ucell[ib]-1 # I'm assuming the tags start counting from 1.
                    element = Element_atypes[atype]
                    symbol = Symbol_atypes[atype]
                    
                    for iele,sym_ele in enumerate(Elements):
                        if sym_ele == element:
                            Num_ele[iele] +=1
                            aaint = Num_ele[iele]%Num_ele_ucell[iele]
                            if aaint == 0:
                                aaint = Num_ele_ucell[iele]
                            aindex = element+str(aaint)

                    fid.write(fom.format(aindex,pos[0],pos[1],pos[2],mol_id[ib]+icell*Nmols_ucell,symbol,element,charges_ucell[ib]))
                icell = icell +1
    fid.write('end')
    fid.close()


def read_lmp_data(in_file,Z_of_type):
    cell0 = io.read(in_file,format='lammps-data')
    Atom_tag = cell0.get_atomic_numbers()
    Atom_No = np.zeros(cell0.get_global_number_of_atoms())
    for (i,Z) in enumerate(Z_of_type):
        iaty = i+1
        Atom_No[Atom_tag==iaty]=Z
    cell0.set_atomic_numbers(Atom_No)
    return cell0


def write_lmp_data(filename,SimCell,molID=[],writeR0=False,atom_style='full',Masses_of_atypes=[]):
    if Masses_of_atypes==[]:
        Masses_of_atypes = np.unique(SimCell.get_masses())
    Number_of_atom_types = len(np.unique(SimCell.get_atomic_numbers()))
    
    Masses = SimCell.get_masses()
    Pos = SimCell.get_positions()
    Charges = SimCell.get_initial_charges()
    tags0 = SimCell.get_tags()

    fid = open(filename,'w')
    
    if writeR0:
        fid2 = open(filename+'.R0','w');
        fid2.write('{}\n'.format(SimCell.get_global_number_of_atoms()))
    
    
    fid.write('LAMMPS data file. \n')
    fid.write('\n')
    fid.write('    {} atoms\n'.format(SimCell.get_global_number_of_atoms()))
    fid.write('\n')
    fid.write('   {} atom types\n'.format(len(np.unique(SimCell.get_atomic_numbers()))))
    fid.write('\n')
    xlo_bound,xhi_bound,ylo_bound,yhi_bound,zlo_bound,zhi_bound,xy,xz,yz = get_lmp_boxbounds(SimCell)
    fid.write('    {:9f}    {:9f} xlo xhi\n'.format(xlo_bound,xhi_bound))
    fid.write('    {:9f}    {:9f} ylo yhi\n'.format(ylo_bound,yhi_bound))
    fid.write('    {:9f}    {:9f} zlo zhi\n'.format(zlo_bound,zhi_bound))
    fid.write('{:6f} {:6f} {:6f} xy xz yz\n'.format(xy,xz,yz))
    fid.write('\n')
    fid.write('Masses\n')
    fid.write('\n')
    for atype in range(Number_of_atom_types):
        fid.write('   {}   {:4f}\n'.format(atype+1,Masses_of_atypes[atype]))
    fid.write('\n')
    
    
    if atom_style == 'charge':
        fid.write('Atoms # charge\n') # use atomic_style full
        fid.write('\n')
        for iat in range(SimCell.get_global_number_of_atoms()):
            for atype in range(Number_of_atom_types):
                if np.abs(Masses[iat] - Masses_of_atypes[atype])<0.01:
                    if np.count_nonzero(tags0)>0:
                        tag =tags0[iat]
                    else:
                        tag = atype+1
                    
            fid.write('{}   {}  {:6f}    {:9f}    {:9f}     {:9f} \n'.format(iat+1,tag,Charges[iat],Pos[iat][0],Pos[iat][1],Pos[iat][2]))
            if writeR0:
                fid2.write('{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {} {}\n'.format(Pos[iat][0],Pos[iat][1],Pos[iat][2],0,0,0,iat+1,tag))
        fid.write('\n')
        
    if atom_style == 'full':
        if molID == []:
            molID = np.ones(SimCell.get_number_of_atoms(),dtype='int32')
        fid.write('Atoms # full\n')
        fid.write('\n')
        for iat in range(SimCell.get_number_of_atoms()):
            for atype in range(Number_of_atom_types):
                if np.abs(Masses[iat] - Masses_of_atypes[atype])<0.01:
                    if np.count_nonzero(tags0)>0:
                        tag =tags0[iat]
                    else:
                        tag = atype+1
            fid.write('{}   {}   {}  {:6f}    {:.9f}    {:.9f}     {:.9f} \n'.format(iat+1,molID[iat],tag,Charges[iat],Pos[iat][0],Pos[iat][1],Pos[iat][2]))
            if writeR0:
                fid2.write('{:9f} {:9f} {:9f} {:9f} {:9f} {:9f} {} {}\n'.format(Pos[iat][0],Pos[iat][1],Pos[iat][2],0,0,0,iat+1,tag))            
        fid.write('\n')   

    fid.close()
    if writeR0:
        fid2.close()


def write_R0(prefix,pcell,scell): # phonopy style R0.
    """ 
    Input objects are prim and supcer cells as phonopy objects.
    """
    Nrepeat = scell.get_supercell_matrix().diagonal()
    fid = open(prefix+'.R0','w')
    
    Nbasis = pcell.get_number_of_atoms()
    Z_basis = pcell.get_atomic_numbers()
    Z_type = np.unique(Z_basis)
    N_types = len(Z_type)
    
    Pos = scell.get_scaled_positions()
    Natoms = scell.get_number_of_atoms()
    
    fid.write('{}\n'.format(Natoms))
    
    iat = 0
    list_basis = np.arange(Nbasis)
    for ityp in range(N_types):
        for ib in list_basis[Z_basis==Z_type[ityp]]:
            for iz in range(Nrepeat[2]):
                for iy in range(Nrepeat[1]):
                    for ix in range(Nrepeat[0]):
                        fid.write('{:9f} {:9f} {:9f} {:9f} {:9f} {:9f} {} {}\n'.format(Pos[iat][0],Pos[iat][1],Pos[iat][2],ix,iy,iz,ib+1,ityp+1))
                        iat += 1
                        
    fid.close()
                        


def write_lmp_dump(filename,Cell_snaps):
    fid = open(filename,'w')
    if type(Cell_snaps) != list:
        Cell_snaps = [Cell_snaps]
        
    Nsnaps = len(Cell_snaps) # Cell_snaps should be a list.
    
    for isnap,cell in enumerate(Cell_snaps):
        Natoms = cell.get_number_of_atoms()
        types = cell.get_tags()
        fid.write('ITEM: TIMESTEP\n')
        fid.write('{}\n'.format(isnap))
        fid.write('ITEM: NUMBER OF ATOMS\n')
        fid.write('{}\n'.format(Natoms))
        fid.write('ITEM: BOX BOUNDS xy xz yz pp pp pp\n')
        xlo_bound,xhi_bound,ylo_bound,yhi_bound,zlo_bound,zhi_bound,xy,xz,yz = get_lmp_boxbounds(cell)
        fid.write('{:6f} {:6f} {:6f}\n'.format(xlo_bound,xhi_bound,xy))
        fid.write('{:6f} {:6f} {:6f}\n'.format(ylo_bound,yhi_bound,xz))
        fid.write('{:6f} {:6f} {:6f}\n'.format(zlo_bound,zhi_bound,yz))
        fid.write('ITEM: ATOMS id type x y z vx vy vz\n')
        pos = cell.get_positions()
        vel = cell.get_velocities()
        for iat in range(Natoms):
            atype = types[iat]+1
            fid.write('{} {} {:6f} {:6f} {:6f} {:6f} {:6f} {:6f}\n'.format(iat,atype,pos[iat][0],pos[iat][1],pos[iat][2],vel[iat][0],vel[iat][1],vel[iat][2]))
    fid.close()
    

# --------------------------- Computing transmisstion matrices for interfaces ------------------------------------#
    

def Compute_MAB_matrix_Gamma(FC2,eigs,molID,groupA,groupB):
    """
    This computes the energy exchange matrix between group A and group B
    MAB(m,n) = Sum(i in A, j in B) [FC2(i,j,a,b)*eig(i,m,a)*eig(j,n,b)]
    n,m are branch indices at Gamma point. 
    """
    
    (Nmodes,Natms,DIM)=eigs.shape
    #print(eigs.shape)
    MAB = np.zeros([Nmodes,Nmodes])
    
    for i in range(Natms):
        for j in range(i,Natms):
            phi_ij = FC2[i,j,:,:]
            phi_ji = phi_ij.transpose()
            if (molID[i] == groupA and molID[j] == groupB) or (molID[i] == groupB and molID[j] == groupA):
                for m in range(Nmodes):
                    emi = np.real(eigs[m,i,:]) # eigs are real at Gamma point
                    emj = np.real(eigs[m,j,:])
                    for n in range(Nmodes):
                        enj = np.real(eigs[n,j,:])
                        eni = np.real(eigs[n,i,:])
                        MAB[m,n] = MAB[m,n] + np.dot(np.matmul(emi,phi_ij),enj) + np.dot(np.matmul(emj,phi_ji),eni)
    
    return MAB

            
                    
# This function is for computing the phonon hybridization ratio between groups
# See the reference: J. Phys. Chem. Lett. 2016, 7, 4744−4750
def calculate_HybridRatio_Groups(atype_groups,phonon,atype_ucell):
    """
        Returns the hybridization ratio of each group at each k point on the dispersion. 
        Will return an array with the shape [# of k-paths, # of k-points, # of branches, # of Groups]
        atype_groups is a list that specify the which atom_type belongs to atom group. e.g [[1,2,3],[4]]
        eigvecs is an eigvenctor.
        atype_ucell specifies the atomic type id for each atom in the unit cell.
    """
    
    eigvecs = phonon.get_band_structure_dict()['eigenvectors']
    
    Npaths,Nk,Ns,Ndof = np.shape(eigvecs)
    Nbasis = int(Ns/3)
    Ngroups = len(atype_groups)
    #print(Ngroups)
    HybridRatio_Groups = np.zeros([Npaths,Nk,Ns,Ngroups])
    
    # build a list of atoms
    groupid_list = np.zeros(Nbasis,dtype='int64')
    for ig,atype_ig in enumerate(atype_groups):
        for ib in range(Nbasis):
            if atype_ucell[ib] in atype_ig:
                groupid_list[ib] = ig
            
    
    for ipath,eigvecs_path in enumerate(eigvecs):
        
        for ik,eigvecs_at_k in enumerate(eigvecs_path):
            
            for js,vec in enumerate(eigvecs_at_k.T):
                eig_ks = np.reshape(vec,[Nbasis,3])
                #Total_sum = np.dot(np.conj(vec),vec)
                Total_sum,group_sum = Sum_dotprod_eigvec_groups(Ngroups,Nbasis,groupid_list,eig_ks)
                            
                for ig in range(Ngroups):
                    #print(Num[ig],Den)
                    HybridRatio_Groups[ipath,ik,js,ig] = group_sum[ig]/Total_sum
    
    
    return HybridRatio_Groups

@njit
def Sum_dotprod_eigvec_groups(Ngroups,Nbasis,groupid_list,eig_ks):
    
    
    group_sum = np.zeros(Ngroups)
    Total_sum = 0
    
    
    
    for ib in range(Nbasis):
        eig_ks_ib = eig_ks[ib]
        dot_prod = np.real(np.dot(np.conj(eig_ks_ib),eig_ks_ib))                           
                
        Total_sum += dot_prod
        ig = groupid_list[ib]
        group_sum[ig] += dot_prod
                
    
    return Total_sum,group_sum



def calc_PartRatio_mesh(phonon):
    # eigvecs should be calculated using set_mesh at gamma point
    qpoints, weights, frequencies, eigvecs = phonon.get_mesh()
    
    (Nqpoints,Natoms_x3,Nmodes)=np.shape(eigvecs)
    Natoms = int(Natoms_x3/3)    
    
    PartRatio = np.zeros([Nqpoints,Nmodes])
    
    for iq,eigvecs_at_q in enumerate(eigvecs):
        for s,vec in enumerate(eigvecs_at_q.T):
            evec = np.reshape(vec,[Natoms,3])
            
            PartRatio[iq,s] = PartRatio_mode(evec)
            

            
    return frequencies,PartRatio

@njit
def PartRatio_mode(evec):
    [Natoms,DIM] = np.shape(evec)
    Den = 0
    Num = 0
    for iat in range(Natoms):
        evec_i = evec[iat]
        Den += (np.conj(evec_i[0])*evec_i[0] + np.conj(evec_i[1])*evec_i[1] + np.conj(evec_i[2])*evec_i[2]).real**2
        Num += (np.conj(evec_i[0])*evec_i[0] + np.conj(evec_i[1])*evec_i[1] + np.conj(evec_i[2])*evec_i[2]).real
        
    Num = Num**2
    
    return Num/Den/Natoms  
            
                    
       
# calculate radial distribution function
def calc_rdf(Cell,rdf_cutoff,rdf_nbins):
    # use ase object for calculating rdf.
    r = neighbour_list('d', Cell, cutoff=rdf_cutoff)
    rdf, bin_edges = np.histogram(r, bins=rdf_nbins, range=(0, rdf_cutoff))
    # normalize by bin volume and total number of atoms
    rdf = rdf / (len(Cell) * 4*np.pi/3 * (bin_edges[1:]**3-bin_edges[:-1]**3))
    # normalize by cell volume
    rdf /= len(Cell)/Cell.get_volume()
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2    
    return bin_centers,rdf        
    
