T=300;
lfactor=1e-10; %m
ATHz2mHz = 100;
ps2s = 1e-12;
lattvec=[3.4061524815891073    0.0000000000000267    0.0000000000000189
         1.7030762407445998    2.9498145781937235    0.0000000000000258
         1.7030762407445998    0.9832715260312809    2.7811118552754275];
Vcell=lfactor^3*abs(det(lattvec));
Ncell = [4 4 4];
kB= 1.3806485279e-23;

Freq_weight_vel_file = '../BAs-qmesh_p4x4x4-irred.groupVel'; 
Tau_file = 'tau_modal_SED.dat';
FWV_Data=importdata(Freq_weight_vel_file);
Tau_Data = importdata(Tau_file);

ks_modes = FWV_Data(:,1:2);
weight_modes = FWV_Data(:,3);
Freq_modes = FWV_Data(:,4);
Vel_modes = FWV_Data(:,5:end)*ATHz2mHz;
Vg2_modes = sum(Vel_modes.^2,2);
Cap_modes = cmode(Freq_modes*2*pi,T);
Tau_modes = Tau_Data(:,3)*ps2s;
Kappa = sum(weight_modes.*Cap_modes.*Vg2_modes.*Tau_modes)/Vcell/prod(Ncell)/3
Kappa_classic = sum(weight_modes.*kB.*Vg2_modes.*Tau_modes)/Vcell/prod(Ncell)/3
% k = zeros(3,3);
% for i=1:3
%     for j=1:3
%         cvt=Cap_modes.*Vel_modes(:,i).*Vel_modes(:,j).*Tau_modes.*weight_modes;
%         k(i,j)=sum(cvt(:))/Vcell/prod(Ncell);
%     end
% end

