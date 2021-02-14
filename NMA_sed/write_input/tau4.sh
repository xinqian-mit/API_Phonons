#!/bin/bash
#SBATCH -J tau4
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -o job-%j.out
#SBATCH -A TG-PHY200027
#SBATCH -p normal
#SBATCH --time 2:00:00
module load intel/18.0.0 fftw3/3.3.8
Phon_COMMAND="/work/07263/xiqi1095/stampede2/NMA/NMA_sed"
${Phon_COMMAND} CONTROL_job4.in  > tau4.out
