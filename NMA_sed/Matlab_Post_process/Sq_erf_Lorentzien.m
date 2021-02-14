function [ F ] = Sq_erf_Lorentzien( w,SED,Paras )
A = Paras(1);
w0 = Paras(2);
Gamma = Paras(3);

L = Lorentzien(w,A,w0,Gamma);

F = sum(((SED)-(L)).^2)/length(SED);


end

