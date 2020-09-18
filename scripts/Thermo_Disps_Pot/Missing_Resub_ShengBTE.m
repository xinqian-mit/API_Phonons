clear;clc;
POSCAR_list=char(importdata('POSCAR_list'));
Nposcars=length(POSCAR_list);
xml_prefix='vasprun-';
vasprun_sufix='.xml';
vasprun_files= [repmat(xml_prefix,Nposcars,1), POSCAR_list,repmat(vasprun_sufix,Nposcars,1)];
job_dir='./jobs/';
system('mkdir Rsub');
system('rm ./Rsub/*');
sh_file_prefix='./Rsub/V_resub_';
wall_time='5:00:00';
Nodes=4;
N_per_sh=2;
Length_Index=2;

imiss=0;
for ipos=1:Nposcars
    str_ipos=num2str(ipos);
    
    if ~exist(['jobs/job_' repmat('0',1,Length_Index-length(str_ipos)),str_ipos],'dir')
        imiss=imiss+1;
        miss_No(imiss)=ipos;
        POSCAR_miss(imiss,:)=POSCAR_list(ipos,:);
    else
       if_error_free=check_vasprun_xml(['jobs/job_' repmat('0',1,Length_Index-length(str_ipos)),str_ipos,'/vasprun.xml']);
       if (~if_error_free)
            imiss=imiss+1;
            miss_No(imiss)=ipos;
            POSCAR_miss(imiss,:)=POSCAR_list(ipos,:);
       end
    end
end

Nmiss=imiss;


str='%s\n';
for i_sh=1:Nmiss/N_per_sh
    minI=(i_sh-1)*N_per_sh+1;
    maxI=i_sh*N_per_sh;
    filename=[sh_file_prefix,POSCAR_miss(minI,end-Length_Index+1:end),'to',POSCAR_miss(maxI,end-Length_Index+1:end),'.sh'];
    fid=fopen(filename,'w');
    fprintf(fid,str,'#!/bin/bash');
    fprintf(fid,'#SBATCH --nodes %g\n',Nodes);
    fprintf(fid,str,'#SBATCH --ntasks-per-node 24');
    fprintf(fid,str,'#SBATCH --qos normal');
    fprintf(fid,str,['#SBATCH --time ' wall_time]);
    
    fprintf(fid,str,'module load intel/17.4 impi/17.3 mkl/17.3');
    fprintf(fid,str,'export OMP_NUM_THREADS=1');
    fprintf(fid,str,'export I_MPI_COMPATIBILITY=4');
    fprintf(fid,str,'mkdir vaspruns_rem');
    
    fprintf(fid,'\n');
    for imiss=minI:maxI
        str_imiss=num2str(miss_No(imiss));
        fprintf(fid,str,['mkdir job_',repmat('0',1,Length_Index-length(str_imiss)),str_imiss]);
        fprintf(fid,str,['cd ./job_',repmat('0',1,Length_Index-length(str_imiss)),str_imiss]);
        fprintf(fid,str,'cp ../INCAR ./');
        fprintf(fid,str,'cp ../POTCAR ./');
        fprintf(fid,str,'cp ../KPOINTS ./');
        fprintf(fid,str,['cp ../POSCARS/',POSCAR_miss(imiss,:),' POSCAR']);
        fprintf(fid,str,'time mpirun /projects/xiqi1095/Software/vasp5.3/vasp.5.3/vasp');
        fprintf(fid,str,['cp vasprun.xml ../vaspruns_rem/vaspruns.' POSCAR_miss(imiss,:) '.xml']);
        fprintf(fid,str,'cd ..');
        
        fprintf(fid,'\n');
    end
end



Nrem=mod(Nmiss,N_per_sh);
if Nrem
    filename=[sh_file_prefix,POSCAR_miss(end+1-Nrem,end-Length_Index+1:end),'to',POSCAR_miss(end,end-Length_Index+1:end),'.sh'];
        fid=fopen(filename,'w');
    
    fprintf(fid,str,'#!/bin/bash');
    fprintf(fid,'#SBATCH --nodes %g\n',Nodes);
    fprintf(fid,str,'#SBATCH --ntasks-per-node 24');
    fprintf(fid,str,'#SBATCH --qos normal');
    fprintf(fid,str,['#SBATCH --time ' wall_time]);
    
    fprintf(fid,str,'module load intel/17.4 impi/17.3 mkl/17.3');
    fprintf(fid,str,'export OMP_NUM_THREADS=1');
    fprintf(fid,str,'mkdir vaspruns_rem');

    fprintf(fid,'\n');
    for j=Nrem-1:-1:0
        str_imiss_rem=num2str(miss_No(Nmiss-j));
        fprintf(fid,str,['mkdir job_',repmat('0',1,Length_Index-length(str_imiss_rem)),str_imiss_rem]);
        fprintf(fid,str,['cd ./job_',repmat('0',1,Length_Index-length(str_imiss_rem)),str_imiss_rem]);
        fprintf(fid,str,'cp ../INCAR ./');
        fprintf(fid,str,'cp ../POTCAR ./');
        fprintf(fid,str,'cp ../KPOINTS ./');
        fprintf(fid,str,['cp ../POSCARS/',POSCAR_miss(end-j,:),' POSCAR']);
        fprintf(fid,str,'time mpirun /projects/xiqi1095/Software/vasp5.3/vasp.5.3/vasp');
        fprintf(fid,str,['cp vasprun.xml ../vaspruns_rem/vaspruns.' POSCAR_miss(end-j,:) '.xml']);
        fprintf(fid,str,'cd ..');
        fprintf(fid,'\n');
    end
    fclose(fid);
end