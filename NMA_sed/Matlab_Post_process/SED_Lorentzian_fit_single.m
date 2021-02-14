%% This script conducts time-domain fitting of modal SED

clear;clc;
close all;
Units2THz=1; %0.0299792458 for cm^-1
iKpoint=0;
iBrch=2;

datafile=['../SED_data/SED_Kpoint',num2str(iKpoint),'_Brch',num2str(iBrch),'.dat'];
eigs_file='../create_struct_lattdyn/NaCl-qmesh3x3x3-irred.eig';
SED_data=importdata(datafile);
Freq_modes=read_eig( eigs_file,Units2THz); % in THz
Freq_har=Freq_modes(iKpoint+1,iBrch+1);
HalfWidth_Fhar = max([2 Freq_har/6+0.1]);
select_omega=( SED_data(:,1)>(Freq_har-HalfWidth_Fhar)*2*pi & (SED_data(:,1)<(Freq_har+HalfWidth_Fhar)*2*pi));
omega=SED_data(select_omega,1);
SED=SED_data(select_omega,2); % time in ps
freq = omega/2/pi;


omega0 = omega(SED==max(SED));
Apeak = max(SED);

co = get(gca,'colororder');
if (Freq_har>eps)
   Paras = [Apeak, omega0,HalfWidth_Fhar];
   J = @(Paras) Sq_erf_Lorentzien(omega,SED,Paras);
   Paras_f = fminunc(J,Paras);
   Amp_f = Paras_f(1);
   omega0_f = Paras_f(2);
   Gamma_f = Paras_f(3);
   tau_f = 1/2/abs(Gamma_f);
   SED_f = Lorentzien(omega,Amp_f,omega0,Gamma_f);
   plot(freq,SED,'o','MarkerFaceColor',co(1,:)); hold on;
   plot(freq,SED_f,'LineWidth',1.5);
   disp(tau_f);
end



