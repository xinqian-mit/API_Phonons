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

In general, one needs to use gcc, gfortran compilers together with openmpi (if you want to parallelize things). 
Currently, this package is based on the following package versions:

ase 3.20.1 
phonopy 2.7.1
lammps (24 Aug 2020)
quippy (The version as of Sep. 17 2020) 
ALM 2.0.0 

To have proper plot of phonon bandstructures, it's also recommended to install latex support. 
For ubuntu just do "apt-get install texlive-latex-recommended"

The python executable bandplot is used to export band.yaml files to text, new version of phonopy-bandplot by phonopy
doesn't have this function anymore, but using bandplot requires phonopy installed.
