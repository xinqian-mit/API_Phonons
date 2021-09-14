# API_Phonons
----
This is a API that interfaces ase, lammps, phonopy, alamode thirdorder.py, Fourthorder.py and quippy for phonon calulations, written
by Xin Qian (CU Boulder, MIT). <br />
<br />
This package is developed to make it easy to extract  force constants and phononic properties of any 
empirical or machine learning potentials, or directly from AIMD data. The API_*.py files contains functions for interfacing 
different packages,and computing phononic properties. This package also provided examples  for computing those different properties including 
harmonic/thirdorder/fourth force constants, or dielectric tensor, or self-consistently compute force constants renormalized at higher temperatures, which even works
for dynamically unstable system with soft phonon modes. <br />
<br />

### Reference
If you find this package useful, please cite one of the following papers: <br /> 
Xin Qian and Ronggui Yang, Phys. Rev. B 98, 224108 (2018) <br />
Xin Qian, Shenyou Peng, Xiaobo Li, Yujie Wei, Ronggui Yang, Materials Today Physics 10, 100140 (2019) <br />

### Installation

In general, one needs to use gcc, gfortran, Cython compilers and parallel libs if necessary. 
Currently, this package is based on the following packages:<br />

ase 3.20.1<br /> 
phonopy 2.7.1 by A. Togo: https://phonopy.github.io/phonopy <br />
lammps (24 Aug 2020) by Sandia National Lab: https://lammps.sandia.gov/doc/Install_tarball.html<br />
quippy & GAP by A. Brtok and Csanyi et al.: https://libatoms.github.io/GAP/installation.html <br /> 
ALM 2.0.0 by Tadano: https://alm.readthedocs.io/en/develop/ <br />
thirorder.py 1.1.0 by Prof. Wu Li and Prof. Mingo: http://www.shengbte.org/downloads <br />
Fourthorder.py by Dr.Tianli Feng and Prof. Xiulin Ruan et al.: https://github.com/FourPhonon/FourPhonon<br />
<br />

Check the mannuals for each package installing/compiling them. 
Other python libraries are also required:
matscipy <br />
numba # This save you from implementing the code in cython. When a function is called many times,
I add the decorator @njit for just-in-time compilation, which would increase the speed by magnitudes.

### Tips for Installation:

#### thirdorder.py and fourthorder.py
Both thirdorder.py & Fourthorder.py needs to be added to PYTHONPATH for import.

#### quippy pakage
It is recommended to compile using gcc and python 3.8 for the quippy packages.
Certain versions of intel compilers would generate segmentation fault for quippy, when call
quippy.potential.Potential objects' calculate function. 

#### phonopy and plot bandstructures
To have proper plot of phonon bandstructures, it's also recommended to install latex support. 
For ubuntu just do "apt-get install texlive-latex-recommended". 

The python executable bandplot is used to export band.yaml files to text, new version of 
phonopy-bandplot by phonopy doesn't have this function anymore, but using bandplot requires 
phonopy installed.

#### compiling lammps for the python interface
When compiling lammps as python library, remember to do the following <br />
<br />
cd lib/python<br />
cp Makefile.lammps.python3 Makefile.lammps<br />
cd ../src<br />
make yes-python<br />
make foo mode=shared<br />
<br />

### NMA_sed
Code and packages for computing spectral energy density from MD trajectories. See the readme.md in the folder for further details.

### Example Scripts

In the folder Example_Scripts/, I provide several example files on how to use this package for phonon simulations:<br />

#### Dielectric_function_NaCl:<br />
I provided a soap based machine learning potential for NaCl. With the LD_quipGap_phonopy_NAC.py computes the 
harmonic force constants (FC2), while thirorder_gap.py computes third order anharmonic force constants (FC3). The 
usage of thirdorder_gap.py is the same as thirdorder_vasp.py for ShengBTE, execute with <br />
<br />
python3 thirdorder_gap.py na nb nc cutoff(nm)|-n GAP_potential_file <br />
<br />
where na nb and nc are the supercell dimensions, followed by cutoff radius or -n where -n is the number of nearest
neighbors. There are also parallelized scripts for fc3 calculation such as thirdorder_gap_mp.py, compute with:<br />
<br />
python3 thirdorder_gap_mp.py na nb nc cutoff(nm)|-n Nprocesses GAP_potential_file <br />
<br />
Similarly, fourthorder force constants can also be evaluated using:
<br />
python3 fourthorder_gap_mp.py na nb nc cutoff(nm)|-n Nprocesses GAP_potential_file <br />

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
<br />
The file RandDisps_SelfConsist_FC2_FC3.py and RandDisps_SelfConsit_FC2_GAP_ALM.py give examples on how to self-consistently compute second 
and thirdorder force constants. I first use the finite displacement method, and obtain the first generation of eigenvectors. Then I used 
the new set of FC2 and FC3, and a new set of eigenvectors. This process is iterated. Convergence can usually be achieved after ~ 5 iterations.
<br />

#### Generate_nacl_subnmGap:<br />
A simple example of how to generate an inteface or gap using ase package <br />

#### RDF:<br />
A simple example of how to compute radial distribution function g(r) <br />

#### SW_aSi_AFModel:<br />
An example of computing modal diffusivity using Allan-Feldman's throy, under testing,currently still relatively slow. The code is partially accelerated using paralllelized numba.  <br />

#### EIM_LattDyn_NaCl:<br />
This gives example how we can interface lammps and phonopy to perform lattice dynamics based on the EIM potential supported
by lammps. Thirdorder.py and Fourthorder.py are implemented with lammps code for computing FC3 and FC4.
<br />
#### SnapsDFT_FCs_ALM:<br />
This example uses Alamode to compute force constants, based on DFT data of energy and forces of random snapshots
The random snapshots are generated based on this paper: Phys. Rev. B 95, 014302, 2017 by Olle Hellman. 
<br />

#### Thermo_Disps_DFT and Thermo_Disps_Pot:<br />
These two folders implemented the code to randomly generate snapshots, using the Olle Hellman's stochastic method, which can be
used as training datasets for TDEP, similar to AIMD snapshots.
<br />
#### Write_eigs:<br />
This output eigenvector files in the GULP format. 


