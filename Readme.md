# API_Phonons
----
This is a API that interfaces ase, lammps, phonopy, alamode thirdorder.py and quippy for molecular simulation, written
by Xin Qian @ MIT. <br />
<br />
This package is developed to make it easy to extract  force constants and phononic properties of any 
empirical or machine learning potentials, or directly from AIMD data. The API_*.py files contains functions for interfacing 
different packages, while in the directory script/, I provided examples  for computing those different properties including 
harmonic/thirdorder force constants, or dielectric tensor, or self-consistently compute force constants renormalized at higher temperatures, which even works
for dynamically unstable system with soft phonon modes. <br />
<br />
If you find this package useful, please cite one of the following papers: <br /> 
Xin Qian and Ronggui Yang, Phys. Rev. B 98, 224108 (2018) <br />
Xin Qian, Shenyou Peng, Xiaobo Li, Yujie Wei, Ronggui Yang, Materials Today Physics 10, 100140 (2019) <br />

### Installation

In general, one needs to use gcc, gfortran compilers together with openmpi (if you want to parallelize things). 
Currently, this package is based on the following package versions:<br />

ase 3.20.1<br /> 
phonopy 2.7.1<br />
lammps (24 Aug 2020)<br />
quippy (The version as of Sep. 17 2020)<br /> 
ALM 2.0.0<br />
thirorder.py 1.1.0 <br />
Fourthorder.py developed Prof. Xiulin Ruan: https://github.com/FourPhonon/FourPhonon<br />
<br />


Both thirdorder.py & Fourthorder.py has been modified to python3 version.
It is recommended to compile using gcc and python 3.8 for the above packages.
Seem intel compilers would generate segmentation fault for quippy, when call
quippy.potential.Potential objects' calculate function. 

To have proper plot of phonon bandstructures, it's also recommended to install latex support. 
For ubuntu just do "apt-get install texlive-latex-recommended". 

Other python libraries include: 
matscipy <br />
numba # This save you from implementing the code in cython. When a function is called many times,
I add the decorator @njit for just-in-time compilation, which would increase the speed.  

The python executable bandplot is used to export band.yaml files to text, new version of 
phonopy-bandplot by phonopy doesn't have this function anymore, but using bandplot requires 
phonopy installed.

When compiling lammps as python library, remember to do the following <br />
<br />
cd lib/python<br />
cp Makefile.lammps.python3 Makefile.lammps<br />
cd ../src<br />
make yes-python<br />
make foo mode=shared<br />
<br />

When installing thirorder.py, also remember to change the first line to #!/user/bin/env python3

### Example Scripts

In the folder scripts/, I provide several example files on how to use this package for phonon simulations:<br />

#### Dielectric_function_NaCl:<br />
I provided a soap based machine learning potential for NaCl. With the LD_quipGap_phonopy_NAC.py computes the 
harmonic force constants (FC2), while thirorder_gap.py computes third order anharmonic force constants (FC3). The 
usage of thirdorder_gap.py is the same as thirdorder_vasp.py for ShengBTE, execute with <br />
<br />
python3 thirdorder_gap.py na nb nc cutoff(nm)|-n <directory of GAP potential file> <br />
<br />
where na nb and nc are the supercell dimensions, followed by cutoff radius or -n where -n is the number of nearest
neighbors. There are also parallelized scripts for fc3 calculation such as thirdorder_gap_mp.py, compute with:<br />
<br />
python3 thirdorder_gap_mp.py na nb nc cutoff(nm)|-n Nprocesses <directory of GAP potential file> <br />
<br />
The jupyter notebook script then computes the dielectric function.  
for perturbated snapshots. 
<br />
#### Dispersion_Zr_hcp_gap:<br />
This is a fairly simple example for computing phonon dispersion of hcp-phase of Zr.
<br />

#### Tdep_bccZr:<br />
This folder shows how temperature stablized phonon dispersion of bcc-Zr, which 
is dynamically unstable at 0K. Also included a script for relaxing cell structures.
quippy potential objects can set as ase calculators using atom.set_calculator(). Then
the cell can be relaxed using optimize module from ase.
 
#### EIM_LattDyn_NaCl:<br />
This gives example how we can interface lammps and phonopy to perform lattice dynamics based on the EIM potential supported
by lammps. Thirdorder.py and Fourthorder.py are implemented with lammps code for computing FC3 and FC4.
<br />
#### SnapsDFT_FCs_ALM:<br />
This example uses Alamode to compute force constants, based on DFT data of energy and forces of random snapshots
The random snapshots are generated based on this paper: Phys. Rev. B 95, 014302, 2017 by Olle Hellman. 
<br />
#### Tdep_potFCs_ALM:<br />
This gives an example how to self-consistently compute second and thirdorder force constants. I first compute the FC2 and FC3
use the finite displacement method, and obtain the first generation of eigenvectors. Then I used Hellman's method to compute
the new set of FC2 and FC3, and a new set of eigenvectors. This process is iterated. Convergence of phonon bandstructure can
usually be achieved after ~ 5 iterations. 
<br />
#### Thermo_Disps_DFT and Thermo_Disps_Pot:<br />
These two folders implemented the code to randomly generate snapshots, using the Olle Hellman's stochastic method, which can be
used as training datasets for TDEP, similar to AIMD snapshots.
<br />
#### Write_eigs:<br />
This output eigenvector files in the GULP format. 
