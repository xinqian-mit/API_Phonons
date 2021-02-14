clear;clc;close all;
Units2THz=1;
eigs_file='../create_struct_lattdyn/NaCl-qmesh3x3x3-irred.eig';
dir_SED = '../SED_data/';



Freq_modes=read_eig( eigs_file,Units2THz);
[Nk,Nbrch] = size(Freq_modes);
co =get(gca,'colororder');

fid=fopen('tau_modal_SED.dat','w');
fom='%.6f\t%.6f\t%.6f\n';

for ik=0:Nk-1
    for ibrch=0: Nbrch-1
        Freq_har=Freq_modes(ik+1,ibrch+1);
        HalfWidth_Fhar = max([0.1 Freq_har/5] );
        
        if (Freq_har>0)
            
            %             for ic=1:Ncase
            dir=dir_SED;
            datafile=[dir,'SED_Kpoint',num2str(ik),'_Brch',num2str(ibrch),'.dat'];
            SED_data=importdata(datafile);
            select_omega=( SED_data(:,1)>(Freq_har-HalfWidth_Fhar)*2*pi & (SED_data(:,1)<(Freq_har+HalfWidth_Fhar)*2*pi));
            omega=SED_data(select_omega,1);
            SED=SED_data(select_omega,2);
%             if (ic==1)
%                 SED=zeros(size(SED_c));
%             end
%             SED=SED+SED; % time in ps
            %             end
            
            omega0 = omega(SED==max(SED));
            Apeak = max(SED);
            Paras = [Apeak, omega0,HalfWidth_Fhar/20];
            J = @(Paras) Sq_erf_Lorentzien(omega,SED,Paras);
            Paras_f = fminunc(J,Paras);
            Amp_f = Paras_f(1);
            omega0_f = Paras_f(2);
            Gamma_f = Paras_f(3);
            tau_f = 1/2/abs(Gamma_f);
            
            hplot=loglog(omega0_f/2/pi,tau_f,'o');
            fprintf(fid,fom,Freq_har,omega0_f/2/pi,tau_f);
            set(hplot,'MarkerFaceColor',co(1,:),'MarkerEdgeColor',co(1,:));
            hold on;
            drawnow;
        else
            fprintf(fid,fom,Freq_har,Freq_har,0.0);
            
        end
    end
end
Freq_plotted = Freq_modes(Freq_modes>0);
minF = min(Freq_plotted);
maxF = max(Freq_plotted);
xlabel('Frequency (THz)');
ylabel('Phonon Lifetime (ps)');
xlim([round(minF/2) round(maxF*2/10)*10]);