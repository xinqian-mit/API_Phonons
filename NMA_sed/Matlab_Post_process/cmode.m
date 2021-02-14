function [ c_mode,s_mode ] = cmode( omega_Trads,T)
% This script calculates the modal heat capacity
% T here is only a scalar. 
%Vcell=lfactor^3*abs(det(lattvec));
% omega, Trad/s

kB= 1.3806485279e-23; % J/K
hbar = 1.0545718e-22; %J*ps

exp_factor=exp(hbar.*(omega_Trads+eps)/kB/T);
c_mode=kB.*(hbar.*omega_Trads./kB./T).^2.*exp_factor./(exp_factor-1).^2;  %(hbar*omega)^2/(kB*T^2)*n0(n0+1), not normalized by volume yet
c_mode(omega_Trads==0)=0;

s_mode=1/2/T.*hbar.*omega_Trads.*coth(hbar.*(omega_Trads+eps)./2./kB/T)-kB.*log(2.*sinh(hbar.*(omega_Trads+eps)/2/kB/T));

end