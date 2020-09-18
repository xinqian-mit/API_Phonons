% THis is to generate .sh files to submit lots of SCF calculations.
clear;clc;
system('mkdir sbatches');
system('rm ./sbatches/*.sh');
cd './POSCARS';
system('for i in 3RD.POSCAR*; do echo $i; done > ../POSCAR_list');
cd ..;

POSCAR_files=char(importdata('POSCAR_list'));
Nposcars=length(POSCAR_files);
Length_Index=2;

xml_prefix='vasprun-';
N_per_sh=5;

walltime='15:00:00';
Nodes=2;

sh_file_prefix='./sbatches/VASP_sub_';

str='%s\n';
for i_sh=1:Nposcars/N_per_sh
    minI=(i_sh-1)*N_per_sh+1;
    maxI=i_sh*N_per_sh;
    filename=[sh_file_prefix,POSCAR_files(minI,end-Length_Index+1:end),'to',POSCAR_files(maxI,end-Length_Index+1:end),'.sh'];
    fid=fopen(filename,'w');
    
    fprintf(fid,str,'#!/bin/bash');
    fprintf(fid,str,['#SBATCH -J ' POSCAR_files(minI,end-Length_Index+1:end),'_',POSCAR_files(maxI,end-Length_Index+1:end)]);
    fprintf(fid,str,['#SBATCH --nodes ',num2str(Nodes)]);
    fprintf(fid,str,'#SBATCH --ntasks-per-node 24');
    fprintf(fid,str,['#SBATCH -o ',POSCAR_files(minI,end-Length_Index+1:end),'to',POSCAR_files(maxI,end-Length_Index+1:end),'-%j.out']);
    fprintf(fid,str,'#SBATCH --qos normal');
    fprintf(fid,str,['#SBATCH --time ',walltime]);
    
    fprintf(fid,str,'module load intel/17.4 impi/17.3 mkl/17.3');
    fprintf(fid,str,'export OMP_NUM_THREADS=1');
    fprintf(fid,str,'export I_MPI_COMPATIBILITY=4');
    fprintf(fid,str,'mkdir vaspruns');
    
    
    for j=minI:maxI
        strj=num2str(j);
        fprintf(fid,str,['mkdir job_',repmat('0',1,Length_Index-length(strj)),strj]);
        
        fprintf(fid,str,['cd ./job_',repmat('0',1,Length_Index-length(strj)),strj]);
        
        fprintf(fid,str,'cp ../INCAR ./');
        fprintf(fid,str,'cp ../POTCAR ./');
        fprintf(fid,str,'cp ../KPOINTS ./');
        fprintf(fid,str,['cp ../POSCARS/',POSCAR_files(j,:),' POSCAR']);
        
        fprintf(fid,str,'time mpirun /projects/xiqi1095/Software/vasp5.3/vasp.5.3/vasp');
        fprintf(fid,str,['cp vasprun.xml ../vaspruns/vaspruns.' POSCAR_files(j,:) '.xml']);
        fprintf(fid,str,'cd ..');
    end
    fclose(fid);
end

Nrem=mod(Nposcars,N_per_sh);
if Nrem
    filename=[sh_file_prefix,POSCAR_files(end+1-Nrem,end-Length_Index+1:end),'to',POSCAR_files(end,end-Length_Index+1:end),'.sh'];
    fid=fopen(filename,'w');
    
    fprintf(fid,str,'#!/bin/bash');
    fprintf(fid,str,['#SBATCH -J ' POSCAR_files(end+1-Nrem,end-Length_Index+1:end),'_',POSCAR_files(end,end-Length_Index+1:end)]);
    fprintf(fid,str,['#SBATCH --nodes ',num2str(Nodes)]);
    fprintf(fid,str,'#SBATCH --ntasks-per-node 24');
    fprintf(fid,str,['#SBATCH -o ',POSCAR_files(end+1-Nrem,end-Length_Index+1:end),'to',POSCAR_files(end,end-Length_Index+1:end),'-%j.out']);
    fprintf(fid,str,'#SBATCH --qos normal');
    fprintf(fid,str,['#SBATCH --time ',walltime]);
    
    fprintf(fid,str,'module load intel/17.4 impi/17.3 mkl/17.3');
    fprintf(fid,str,'export OMP_NUM_THREADS=1');
    fprintf(fid,str,'export I_MPI_COMPATIBILITY=4');
    fprintf(fid,str,'mkdir vaspruns');
    
    for j=Nrem-1:-1:0
        strremj=num2str(Nposcars-j);
        fprintf(fid,str,['mkdir job_',repmat('0',1,Length_Index-length(strremj)),strremj]);
        
        fprintf(fid,str,['cd ./job_',repmat('0',1,Length_Index-length(strremj)),strremj]);
        
        fprintf(fid,str,'cp ../INCAR ./');
        fprintf(fid,str,'cp ../POTCAR ./');
        fprintf(fid,str,'cp ../KPOINTS ./');
        fprintf(fid,str,['cp ../POSCARS/',POSCAR_files(end-j,:),' POSCAR']);
        fprintf(fid,str,'time mpirun /projects/xiqi1095/Software/vasp5.3/vasp.5.3/vasp');
        fprintf(fid,str,['cp vasprun.xml ../vaspruns/vaspruns.' POSCAR_files(end-j,:) '.xml']);
        fprintf(fid,str,'cd ..');
        
        %fprintf(fid,str,['cp vasprun.xml ../vaspruns/',xml_prefix,POSCAR_files(end-j,:),'.xml']);
    end
    fclose(fid);
end
