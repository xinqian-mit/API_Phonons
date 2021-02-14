clear;clc;
system('rm *.sh *.dat');
nbasis=8;
natyps=2;
mb=[22.989 35.45];
dt_dump=0.0005; %in ps
NSegs= 5;

Ndump=2500000; % length of trajectory dumping


Nks_per_job=24;

Nk=4;
Nbrch=3*nbasis;
info_prefix=['CONTROL_job'];
sufix='.in';
R0_file='NaCl_333.R0';
R_vel_file='../R_vel.dat';
eig_file='NaCl-qmesh3x3x3-irred.eig';
str='%s\n';
PARTITION='normal';
proj_account = 'TG-PHY200027';
N_nodes=1; % Only serial code. 
walltime='2:00:00';%put space here if don't want to specify a walltime
%% write kpoints into different jobs
remain=mod(Nk*Nbrch,Nks_per_job);
Njob=floor(Nk*Nbrch/Nks_per_job)+min(1,remain);
ikscalc=0;
for ik=0:Nk-1
   for iv=0:Nbrch-1
       ikscalc=ikscalc+1;
       K_index(ikscalc,:)=[ik,iv];
   end
end


for ijob=1:Njob

        info_filename=[info_prefix,num2str(ijob),sufix];
        fid=fopen(info_filename,'w');
        fprintf(fid,str,'pos_file:');
        fprintf(fid,'%s\n',R0_file);
        fprintf(fid,str,'R_vel_file:');
        fprintf(fid,'%s\n',R_vel_file);
        fprintf(fid,str,'eig_file:');
        fprintf(fid,'%s\n',eig_file);
        fprintf(fid,'%s %g\n','nbasis=',nbasis);
        fprintf(fid,'%s %g\n','natyps=',natyps);
        fprintf(fid,'%s','mb= ');
        for iatyps=1:natyps
            fprintf(fid,'%.3f ',mb(iatyps));
        end
        fprintf(fid,'\n');
        fprintf(fid,'%s %d\n','NSegs=',NSegs);
        fprintf(fid,'%s %f\n','dt_dump=',dt_dump);
        fprintf(fid,'%s %d\n','Ndump=',Ndump);
        
        fprintf(fid,'%s %g\n','Nks_per_job=',Nks_per_job);%please make Nk_per_thrd a factor of Nk
        
        if (ijob<=floor(Nk*Nbrch/Nks_per_job))
            for imode = 1+Nks_per_job*(ijob-1):Nks_per_job*ijob
                fprintf(fid,'%g %g\n',K_index(imode,:));
            end
        else
            for irem=remain:-1:0
              fprintf(fid,'%g %g\n',K_index(Nk*Nbrch-irem,:)); 
            end

        end
        fclose(fid);
        

end

    
    
%% write bash
bsh_prefix=['tau'];

    for i=1:Njob
        info_filename=[info_prefix,num2str(i),sufix];
        bsh_file=[bsh_prefix,num2str(i),'.sh'];
        outfile=[bsh_prefix,num2str(i),'.out'];
        fid=fopen(bsh_file,'w');
        fprintf(fid,str,'#!/bin/bash');
        fprintf(fid,'%s %s%g\n','#SBATCH -J', bsh_prefix,i);
        fprintf(fid,'%s %g\n','#SBATCH -N',N_nodes);
        fprintf(fid,str,'#SBATCH --ntasks-per-node 1');
        fprintf(fid,str,'#SBATCH -o job-%j.out');
        fprintf(fid,str,['#SBATCH -A ' proj_account]);
        fprintf(fid,'%s %s\n','#SBATCH -p',PARTITION);
        if (~strcmp(walltime,' '))
        fprintf(fid,'%s %s\n','#SBATCH --time',walltime);
        end
        fprintf(fid,str,'module load intel/18.0.0 fftw3/3.3.8');
        fprintf(fid,str,'Phon_COMMAND="/work/07263/xiqi1095/stampede2/NMA/NMA_sed"');
        fprintf(fid,'%s %s %s %s\n','${Phon_COMMAND}',info_filename,' >',outfile);
    end



