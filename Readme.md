# API Phonons
----
Develped by **Xin Qian** (CU Boulder, MIT, HUST, current contant: xinqian21@hust.edu.cn).  

**API Phonons** is developed to perform complex phonon calculations when multiple packages are used, including extracting force constants and phononic properties of any empirical or machine learning potentials, calculating thermal conductivity using the Kubo's linear response theory, and computing pump-probe thermal responses using mode-resolved phonon properties. The API_xxx.py files contains modules and functions for interfacing different packages such as phonopy, lammps, phono3py, ShengBTE *etc*. All function names and modules should be self-explanatory.

## References
If you find this package useful, please cite one of the following papers:  
> Xin Qian and Ronggui Yang, Phys. Rev. B 98, 224108 (2018)  
> Xin Qian, Shenyou Peng, Xiaobo Li, Yujie Wei, Ronggui Yang, Materials Today Physics 10, 100140 (2019)  
## Installation

In general, one needs to use gcc, gfortran, Cython compilers and parallel libs if necessary. 
Currently, **API Phonons** is built based the following packages:<br />

> **ase** >3.20.1: https://wiki.fysik.dtu.dk/ase/  
> **phonopy** >2.7.1 by A. Togo: https://phonopy.github.io/phonopy  
> **lammps** by Sandia National Lab: https://lammps.sandia.gov/doc/Install_tarball.html  
> **quippy & GAP** by A. Brtok and Csanyi et al.: https://libatoms.github.io/GAP/installation.html  
> **ALM 2.0.0** by Tadano: https://alm.readthedocs.io/en/develop/   
> **thirorder.py** by Prof. Wu Li and Prof. Mingo: http://www.shengbte.org/downloads  
> **Fourthorder.py** by Dr.Tianli Feng and Prof. Xiulin Ruan et al.: https://github.com/FourPhonon/FourPhonon  

Please check the mannuals for each package installing/compiling them. 
Other python libraries such as matscipy and numba are also required, which is usually provided if you use anaconda.

## Tips for Installation:

#### thirdorder.py and fourthorder.py
Both thirdorder.py & Fourthorder.py needs to be added to PYTHONPATH for import.

#### quippy pakage
It is recommended to compile using gcc and python (>3.8) for the quippy packages.
Certain versions of intel compilers would generate segmentation fault for quippy, when call
quippy.potential.Potential objects' calculate function. 

#### phonopy and phonon dispersions
To have proper plot of phonon bandstructures, it's also recommended to install latex support. 
For ubuntu linux system just do:

```
apt-get install texlive-latex-recommended
```

The python executable bandplot is used to export band.yaml files to text, new version of 
phonopy-bandplot by phonopy doesn't have this function anymore, but using bandplot requires 
phonopy installed.

#### lammps with the python interface
When compiling lammps as python library, remember to do the following:  

```
cd lib/python
cp Makefile.lammps.python3 Makefile.lammps
cd ../src
make yes-python
make foo mode=shared
```
Then add lammps to the envoronment variable PYTHONPATH.


## Example Scripts

In the folder Example_Scripts/, I provide several example files on how to use this package for phonon simulations:<br />


#### Dispersion_Zr_hcp_gap:  
This is a fairly simple example for computing phonon dispersion of hcp-phase of Zr.  

I provided a soap based machine learning potential for hcp-Zr. With the LD_quipGap_phonopy_NAC.py computes the 
harmonic force constants (FC2), while thirorder_gap.py computes third order anharmonic force constants (FC3). The 
usage of thirdorder_gap_mp.py is the same as thirdorder_vasp.py for ShengBTE, execute with:  


```
python thirdorder_gap_mp.py [na] [nb] [nc] [cutoff(nm)]|-[n] [Nprocesses] [GAP_potential_file]
```

where the parameters in [] need to be specified. Parameters named na nb and nc are the supercell dimensions, followed by cutoff radius or -n where -n is the number of nearest
neighbors.

Similarly, fourthorder force constants can also be extracted using:

```
python fourthorder_gap_mp.py [na] [nb] [nc] [cutoff(nm)]|-[n] [Nprocesses] [GAP_potential_file]
```
The jupyter notebook script then computes the dielectric function.  
for perturbated snapshots.   


#### Tdep_bccZr: 
This folder shows how temperature stablized phonon dispersion of bcc-Zr, which 
is dynamically unstable at 0K. Also included a script for relaxing cell structures.
quippy potential objects can set as ase calculators using atom.set_calculator(). Then
the cell can be relaxed using optimize module from ase.  

The file RandDisps_SelfConsist_FC2_FC3.py and RandDisps_SelfConsit_FC2_GAP_ALM.py give examples on how to self-consistently compute second 
and thirdorder force constants. I first use the finite displacement method, and obtain the first generation of eigenvectors. Then I used 
the new set of FC2, FC3, eigenvectors to generate snapshots. This process is iterated. Convergence can usually be achieved after ~ 5 iterations.  

#### RDF:  
A simple example of how to compute radial distribution function g(r).  


#### SW_aSi_AFModel:  
An example of computing amorphous silicon thermal conductivity using Allen-Feldman Theory.  


#### EIM_AnHar_QHGK_NaCl:  
Compute the thermal conductivity of EIM NaCl using quasi-harmonic Kubo relations from quantum mechanical linear response theories. Phono3py is used to compute phonon linewidths. The velocity matrix operators are computed the same way in the Allen-Feldman model.

#### QHGK_BTE_CsPbBr:
Compute thermal conductivity using quasi-harmonic Kubo formalism, with both the diagonal (phonon quasiparticles) and the off-diagonal (inter-band coherent tunneling) contributions. We have used orthorombic CsPbBr3 as an example. 

#### EIM_LattDyn_NaCl:
This gives example how we can interface lammps and phonopy to perform lattice dynamics based on the EIM potential supported
by lammps. Thirdorder.py and Fourthorder.py are implemented with lammps code for computing FC3 and FC4.

#### SnapsDFT_FCs_ALM:  
This example uses Alamode to compute force constants, based on DFT data of energy and forces of random snapshots assuming a Gaussian-like vibration around the equilibrium position. 
The random snapshots are generated based on this paper: Phys. Rev. B 95, 014302, 2017 by Olle Hellman.   

#### Thermo_Disps_DFT and Thermo_Disps_Pot:  
These two folders implemented the code to randomly generate snapshots, using the Olle Hellman's stochastic method, which can be
used as training datasets for TDEP, similar to AIMD snapshots.  

#### Write_eigs:  
This script is used to compute eigenvectors and write them in the GULP format.   

