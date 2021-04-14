# thirdorder #

The purpose of the thirdorder scripts is to help users of [ShengBTE](https://bitbucket.org/sousaw/shengbte) and [almaBTE](http://www.almabte.eu) create FORCE\_CONSTANTS\_3RD files in an efficient and convenient manner. More specifically, it performs two tasks:

1) It resolves an irreducible set of atomic displacements from which to compute the full anharmonic interatomic force constant (IFC) matrix. The displaced supercells are saved to input files that can be fed to first-principles DFT codes for calculating the forces arising from the atomic displacements. Currently supported DFT codes are VASP (thirdorder_vasp.py), Quantum ESPRESSO (thirdorder_espresso.py). and CASTEP (thirdorder_castep.py).

2) From the output files created by the DFT code, thirdorder reconstructs the full IFC matrix and writes it in the right format to FORCE\_CONSTANTS\_3RD.

# Compilation #

thirdorder is a set of Python scripts. It was developed using Python 2.7.3, but should work with slightly older versions. In addition to the modules in Python's standard library, the numpy and scipy numerical libraries are required. Moreover, this script relies on a module, thirdorder\_core, which is written in Cython. Thus, in spite of Python being an interpreted language, a compilation step is needed. Note that in addition to the .pyx source we also distribute the intermediate .c file, so Cython itself is not needed. The only requirements are a C compiler, the Python development package and Atsushi Togo's [spglib](http://spglib.sourceforge.net/).

Compiling can be as easy as running

```bash
./compile.sh
```

However, if you have installed spglib to a nonstandard directory, you will have to perform some simple editing on setup.py so that the compiler can find it. Please refer to the comments in that file.

# Usage #

After a successful compilation, the directory will contain the compiled module thirdorder\_core.so, thirdorder_common.py and DFT-code specific interfaces (e.g. thirdorder_vasp.py). All are needed to run thirdorder. You can either use them from that directory (maybe including it in your PATH for convenience) or copying the .py files to a directory in your PATH and thirdorder\_core.so to any location where Python can find it for importing.

# Running thirdorder with VASP #

Any invocation of thirdorder_vasp.py requires a POSCAR file with a description of the unit cell to be present in the current directory. The script uses no other configuration files, and takes exactly five mandatory command-line arguments:

```bash
thirdorder_vasp.py sow|reap na nb nc cutoff[nm/-integer]
```

The first argument must be either "sow" or "reap", and chooses the operation to be performed (displacement generation or IFC matrix reconstruction). The next three must be positive integers, and specify the dimensions of the supercell to be created. Finally, the "cutoff" parameter decides on a force cutoff distance. Interactions between atoms spaced further than this parameter are neglected. If cutoff is a positive real number, it is interpreted as a distance in nm; on the other hand, if it is a negative integer -n, the maximum distance among n-th neighbors in the supercell is automatically determined and the cutoff distance is set accordingly.

The following POSCAR describes the relaxed geometry of the primitive unit cell of InAs, a III-V semiconductor with a zincblende structure:

```
InAs
   6.00000000000000
     0.0000000000000000    0.5026468896190005    0.5026468896190005
     0.5026468896190005    0.0000000000000000    0.5026468896190005
     0.5026468896190005    0.5026468896190005    0.0000000000000000
   In   As
   1   1
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.2500000000000000  0.2500000000000000  0.2500000000000000
```

Let us assume that such POSCAR is in the current directory and that thirdorder_vasp.py is in our PATH. To generate an irreducible set of displacements for a 4x4x4 supercell and up-to-third-neighbor interactions, we run

```bash
thirdorder_vasp.py sow 4 4 4 -3
```

This creates a file called 3RD.SPOSCAR with the undisplaced supercell coordinates and 144 files with names following the pattern 3RD.POSCAR.*. It is the latter that need to be input to VASP. This step is completely system-dependent, but suppose that in ~/vaspinputs we have the required INCAR, POTCAR and KPOINTS files as well as a runvasp.sh script that can be passed to qsub. We can run a command sequence like:

```bash
for i in 3RD.POSCAR.*;do
   s=$(echo $i|cut -d"." -f3) &&
   d=job-$s &&
   mkdir $d &&
   cp $i $d/POSCAR &&
   cp ~/vaspinputs/INCAR ~/vaspinputs/POTCAR ~/vaspinputs/KPOINTS $d &&
   cp ~/vaspinputs/runvasp.sh $d &&
   (cd $d && qsub runvasp.sh)
done
```

Some time later, after all these jobs have finished successfully, we only need to feed all the vasprun.xml files in the right order to thirdorder_vasp.py, this time in reap mode:

```bash
find job* -name vasprun.xml|sort -n|thirdorder_vasp.py reap 4 4 4 -3
```

If everything goes according to plan, a FORCE\_CONSTANTS\_3RD file will be created at the end of this run. Naturally, it is important to choose the same parameters for the sow and reap steps.

# Running thirdorder with Quantum ESPRESSO #

The invocation of thirdorder_espresso.py requires two files:

1) an input file of the unit cell with converged structural parameters

2) a template input file for the supercell calculations. The template file is a normal QE input file with some wildcards

The following input file GaAs.in describes the relaxed geometry of the primitive unit cell of GaAs, a III-V semiconductor with a zincblende structure

```
&CONTROL
 calculation='scf',
 prefix='gaas',
 restart_mode='from_scratch',
 tstress = .true.,
 tprnfor = .true.,
/
&SYSTEM
 ibrav=0,
 nat=2,
 ntyp=2,
 ecutwfc=48
 ecutrho=384
/
&ELECTRONS
 conv_thr=1.d-12,
/
ATOMIC_SPECIES
 Ga  69.723    Ga.pbe-dnl-kjpaw_psl.1.0.0.UPF
 As  74.92160  As.pbe-dn-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
 Ga 0.00 0.00 0.00
 As 0.25 0.25 0.25
K_POINTS automatic
11 11 11 0 0 0
CELL_PARAMETERS angstrom
   0.000000000   2.857507756   2.857507756
   2.857507756   0.000000000   2.857507756
   2.857507756   2.857507756   0.000000000
```

thirdorder_espresso.py supports the following QE input conventions for structural parameters:

1) ibrav = 0 together with CELL_PARAMETERS (alat | bohr | angstrom)

2) ibrav != 0 together with celldm(1)-celldm(6)

For ATOMIC_POSITIONS, all QE units are supported (alat | bohr | angstrom | crystal). Simple algebraic expressions for the positions are supported in similar fashion to QE. Please note that ibrav = 11..14  have not been tested so far with thirdorder_espresso.py (please report if you run these cases successfully or run into problems). Cases ibrav = -5, -9, -12, -13, and 91 are not currently implemented (but those structures can be defined via ibrav = 0 instead)

The following supercell template GaAs_sc.in is used for creating the supercell input files (note the ##WILDCARDS##):

```
&CONTROL
  calculation='scf',
  prefix='gaas',
  tstress = .true.,
  tprnfor = .true.,
  outdir = 'tmp_\#\#NUMBER\#\#'
/
&SYSTEM
  ibrav=0,
  nat=\#\#NATOMS\#\#,
  ntyp=2,
  ecutwfc=48
  ecutrho=384
/
&ELECTRONS
  conv_thr=1.d-12,
/
ATOMIC_SPECIES
 As  74.92160  As.pbe-dn-kjpaw_psl.1.0.0.UPF
 Ga  69.723    Ga.pbe-dnl-kjpaw_psl.1.0.0.UPF
\#\#COORDINATES\#\#
K_POINTS gamma
\#\#CELL\#\#
```

Please note that if Gamma-point k-sampling is used for the supercells, it is computationally much more efficient to apply "K_POINTS gamma" instead of "K_POINTS automatic" with the mesh set to "1 1 1 0 0 0". SCF convergence criterion conv_thr should be set to a tight value and parameters tstress and tprnfor are required so that thirdorder can extract the forces from the output file.

Thirdorder uses no other configuration files, and requires seven mandatory command-line arguments to create the supercell inputs with the "sow" operation:

```bash
thirdorder_espresso.py unitcell.in sow na nb nc cutoff[nm/-integer] supercell_template.in
```

Please see the above description for VASP for the explanation of the parameters na, nb, nc, and cutoff. For the present GaAs example, we execute:

```bash
thirdorder_espresso.py GaAs.in sow 4 4 4 -3 GaAs_sc.in
```

The command creates a file called BASE.GaAs_sc.in with the undisplaced supercell coordinates and 144 files with names following the pattern DISP.GaAs_sc.in.NNN The DISP files should be executed with QE. This step is completely system-dependent, but some practical suggestions can be extracted from the VASP example above.

After all the jobs have finished successfully, we only need to feed all the output files in the right order to thirdorder_espresso.py, this time in reap mode (now using only six arguments, the supercell argument is not used here):

```bash
find . -name 'DISP.GaAs_sc.in*out' | sort -n | thirdorder_espresso.py GaAs.in reap 4 4 4 -3
```

If everything goes according to plan, a FORCE_CONSTANTS_3RD file will be created at the end of this run. Naturally, it is important to choose the same parameters for the sow and reap steps.

# Running thirdorder with CASTEP #

Any invocation of thirdorder_castep.py requires a CELL and PARAM file with a description of the unit cell and parameters to be present in the current directory. The script uses no other configuration files, and takes 
exactly six mandatory command-line arguments:

```bash
thirdorder_castep.py sow|reap na nb nc cutoff[nm/-integer] <seedname>
```

The first argument must be either "sow" or "reap", and chooses the operation to be performed (displacement generation or irreducible force constant (IFC) matrix reconstruction). The next three must be positive integers, 
and specify the dimensions of the supercell to be created. The "cutoff" parameter specifies the force cutoff distance in nanometres; interactions between atoms further apart than this parameter are neglected. If cutoff is 
a negative integer -n, the cutoff is set automatically to the maximum distance of the n-th nearest neighbours in the supercell, e.g. if it is set to -3, the 3rd nearest-neighbour distances will be computed, and the cutoff set to the
 largest value. Finally, <seedname> is the file prefix for CASTEP's input/output files.

The following CELL file describes the relaxed geometry of the primitive unit cell of InAs, a III-V semiconductor with a zincblende structure:

```
%BLOCK LATTICE_CART
     0.000000000000000    3.015881337714003    3.015881337714003
     3.015881337714003    0.000000000000000    3.015881337714003
     3.015881337714003    3.015881337714003    3.015881337714003
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
In  0.0000000000000000  0.0000000000000000  0.0000000000000000
As  0.2500000000000000  0.2500000000000000  0.2500000000000000
%ENDBLOCK POSITIONS_FRAC

symmetry_generate
kpoints_mp_grid 5 5 5
```

Let us assume that such a CELL file is in the current directory along with an appropriate PARAM file, and that thirdorder_castep.py is in our PATH. To generate an irreducible set of displacements for a 4x4x4 supercell and 
up to third-nearest-neighbour interactions, we run:

```bash
thirdorder_castep.py sow 4 4 4 -3 InAs
```

This creates an InAs-3RD directory (`<seedname>`-3RD in the general case) with the undisplaced supercell coordinates and 144 subdirectories with names following the pattern job-*, which contain supercells with small perturbations 
to the atomic positions. Each job is a separate calculation which needs to be input to CASTEP. This step is completely system-dependent. As an example, on a given system the user could run the jobs in series with a command sequence like:

```bash
for i in {000..144}
do
 cd InAs-3RD/job-$i
 aprun -n ${n} castep.mpi InAs
 cd -
 echo "job-$i done" >> jobs_done.txt
done
```

It is necessary to complete all jobs in `<seedname>`-3RD directory before proceeding to the REAP step. After the jobs have completed successfully, the output files have to be collated and passed to thirdorder_castep.py, this 
time in REAP mode. The general syntax is:

```bash
find <seedname>-3RD/job* -name <seedname>.castep | sort -n| thirdorder_castep.py reap nx ny nz cutoff seedname
```

For the InAs example, this would be:

```bash
find InAs-3RD/job* -name InAs.castep | sort -n| thirdorder_castep.py reap 4 4 4 -3 InAs
```

If everything goes well, a FORCE\_CONSTANTS\_3RD file will be created at the end of this run. Naturally, it is important to choose the same parameters (nx, ny, nz, cutoff) for the sow and reap steps. Use this 
FORCE\_CONSTANTS\_3RD file along with FORCE\_CONSTANTS\_2ND and CONTROL to perform a ShengBTE run. The [CASTEP2ShengBTE script](https://github.com/ganphys/castep2shengbte) can be useful for generating FORCE\_CONSTANTS\_2ND and CONTROL files from CASTEP calculations.

### Limitations of the CASTEP interface: 
- Spin-polarised calculations are not supported at the moment. Spin values will not be included in the supercell files.
- The initial <seedname>.cell file MUST be in the following format:
	  Lattice parameter, Cell contents AND THEN everything else.
- Only fractional coordinates are supported. Use only fractional coordinates.

### Hints and tips for CASTEP calculations:

- Use `write_checkpoint: none` in the `<seedname>`.param file. Otherwise, the process of writing hundreds of checkpoint files to the hard drive will slow down the calculation process.
 
- It is possible to reuse a single checkpoint file for each of the runs. This should save you a couple of hours. For that purpose, generate a checkpoint file from one of the runs and place the file in the root directory where your input files are placed. Then add `reuse : ../../seedname.check` to your `<seedname>`.param file in the root directory and either run once again thirdorder_castep.py in SOW mode paste the edited `<seedname>`.param file to all subdirectories or copy and paste it manually.

- If you don't want to generate the pseudopotentials at the start of each run, you can add the following block to the end of the <seedname>.cell in the root directory:

```
%BLOCK SPECIES_POT
In ../../In_C17_PBE_OTF.usp
As ../../As_C17_PBE_OTF.usp
%ENDBLOCK SPECIES_POT
```

Please note that you need to edit the elements and the name of the pseudopotentials in accordance to your system.