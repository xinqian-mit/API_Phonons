# Allen-Feldman Model using API_Phonons
----
This example shows how to compute thermal conductivity of amorphous silicon using Allen-Felman model based on API_Phonons. <br />
<br />

### Relax_Cell.py
This script first relaxes the initial structure POSCAR_512 with 512 silicon atoms. <br /> 

### calc_write_Eigs.py

This script computes the force constants and eigenvectors, which are exported with phonopy forat and GULP format respectively.<br />

### AF_aSi.ipynb and AF_aSi.py

Based on the force constants, these ipynb and py shows how diffusivities of vibrational modes are computed, and thermal conductivity is calculated.<br />

