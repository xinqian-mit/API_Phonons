# API_Phonons
----
This is a API that interfaces ase, lammps, phonopy, alamode thirdorder.py and quippy for molecular simulation. Written
by Xin Qian @ MIT.

To use this package, one needs to install phonoy, thirdorder.py lammps python interface, quippy and ALM. One could 
check the tutorials for those packages on how to install them. 

This package is written to interface different phonon simulation tools, making it convenient to extract force constants
of any empirical or machine learning potentials, or from AIMD data. The API_*.py files contains functions for interfacing
different packages, while in the directory script/, I provided examples for computing those different properties including
harmonic/thirdorder force constants, or dielectric tensors. Utilizing Alamode, it can also self-consistently compute force
constants renormalized at higher temperatures. 

### Installation

In general, one needs to use gcc, gfortran compilers together with openmpi (if you want to parallelize things). 
Currently, this package is based on the following package versions:<br />

ase 3.20.1<br /> 
phonopy 2.7.1<br />
lammps (24 Aug 2020)<br />
quippy (The version as of Sep. 17 2020)<br /> 
ALM 2.0.0<br />
thirorder.py 1.1.0 <br />
<br />

It is recommended to compile using gcc and python 3.8 for the above packages.
Seem intel compilers would generate segmentation fault for quippy, when call
quippy.potential.Potential objects' calculate function. 

To have proper plot of phonon bandstructures, it's also recommended to install latex support. 
For ubuntu just do "apt-get install texlive-latex-recommended". Also remember to
install joblib and matscipy.

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
usage of thirdorder_gap.py is the same as thirdorder_vasp.py for ShengBTE. The jupyter notebook script then computes and plots 
the dielectric function. 
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
by lammps.
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
These two folders implemented the code to randomly generate snapshots, using the Hellman's method
<br />
#### Write_eigs:<br />
This output eigenvector files in the GULP format. 
