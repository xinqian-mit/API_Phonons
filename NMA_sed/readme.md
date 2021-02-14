This code is written by Xin Qian @ Univ. of Colorado Boulder. Contact: xin.qian@Colorado.edu. <br />
Compile on ubuntu: <br />
g++ --std=c++11 NMA_sed.cpp -o NMA_sed -lfftw3 -lm <br />
on TACC: <br />
icpc --std=c++11 NMA_sed.cpp -o NMA_sed -I$TACC_FFTW3_INC -Wl,-rpath,$TACC_FFTW3_LIB  -L$TACC_FFTW3_LIB -lfftw3 <br />
<br />
<br />


To run phonon normal mode analysis (NMA). Here are the steps to take.<br />
<br />
1. Run lattice dynamics with gulp by listing the allowed k-points and the output the eigenvectors.
   Noted that the gulp output frequency in units of 1/cm, you need to set the convertion coefficients to convert that to either
   LJ units or THz. Recompilation of the code is needed when you do that. 
<br />
2. Run the lammps and output trajectroy of position a and velocity by using the lmps command:  
<br />
dump Rv all custom ${dv_intv} R_vel_disp.dat x y z vx vy vz # dv_intv is the interval of sample the trajectories
dump_modify Rv sort id                                      # Output order is the atomic id. 
<br />
3. The info_file listed basic information taken by the program, formmated in this way: (Do remove the comments when running)
<br />
<br />
pos_file:<br />
./R0_disp_Ar_uLJ101010.dat // The file that listed the equilibrium position, unit cell index and the basis index in unit cell  <br />
R_vel_file:<br />
./R_vel_disp_1.dat         // The trajectroy file  <br />
eig_file:<br />
./Ar_conv.eig              // The eigenvector file generated in gulp format<br />
nbasis= 4                  // number of basis in the unitcell<br />
natyps= 1                  // number of atom types<br />
natom_atyp= 4              // number of atoms in each atom type. Do make sure the order of basis (like vasp)<br />
mb= 1.0                    // masses of each atom type <br />
Nsegs= 4                   // Number of segs to compute the SED<br />
dt_dump= 0.05              // time interval between the dump of trajectory in lammps<br />
Ndump= 400                 // total length of the dump in trajectory file<br />
Nks_per_job= 1             // specify the number of modes that you want to consider<br />
8 2                        // index of wavevector and branch referring to the .eig file. <br />
<br />
4. The needed files are: R_vel_file, pos_file, eig_file and the info_file<br />
<br />
5. use write_info.m to generate input files and submission bash files.<br />
<br />
6. Output files are SED_Kpoint[k]_Brch[s].dat.<br />



