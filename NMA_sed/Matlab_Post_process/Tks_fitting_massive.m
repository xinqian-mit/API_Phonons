%% This script conducts time-domain fitting of modal kinetic energy
% This fitting is suggested if specified equilibrium position is no
% accurate.

clear;clc;
cminv2THz=0.0299792458;
NMode=25; % total 21
NBrch=12; %total 24
tend=50;
Ncase=1;

co=[     0    0.4470    0.7410
        0.8500    0.3250    0.0980
        0.9290    0.6940    0.1250
        0.4940    0.1840    0.5560
        0.4660    0.6740    0.1880
        0.3010    0.7450    0.9330
        0.6350    0.0780    0.1840]; % default colors


eigs_file=['rec_Graphene_1051.eig'];
Freq_modes=read_eig( eigs_file,cminv2THz); % in THz


    
fid=fopen('Grp_ave_tau.dat','w');
fom='%.6f\t%.6f\n';
for iMode=0: NMode-1
    for iBrch=0: NBrch-1
        
        for ic=1:Ncase
            dir=['./c',num2str(ic),'/'];
            datafile=[dir,'tau_Mode',num2str(iMode),'_Brch',num2str(iBrch),'.dat'];
            tau_data=importdata(datafile);
            trange=(tau_data(:,1)<tend);
            t=tau_data(trange,1);
            if (ic==1)
                Tks=zeros(length(t),1);
            end
            Tks=Tks+tau_data(trange,3)/Ncase; % time in ps
        end
        Freq_har=Freq_modes(iMode+1,iBrch+1);
        Paras=[10,Freq_har]; % fitting parameters
        J=@(Paras)Sq_erf(t,Tks,Paras);
        [Parasf]=fminunc(J,Paras);
        tau=Parasf(1); Freq_anh=Paras(2);
        hplot=loglog(Freq_har,tau,'o');
        fprintf(fid,fom,Freq_har,tau);
        set(hplot,'MarkerFaceColor',co(1,:),'MarkerEdgeColor',co(1,:));
        hold on;
        drawnow;
    end
end
    

axis([0.1 65 0.01 20]);
fclose(fid);


xlabel('Frequency (THz)');
ylabel('Lifetime, \tau (ps)');
