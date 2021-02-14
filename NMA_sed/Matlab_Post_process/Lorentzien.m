function [ L ] = Lorentzien( w,Amp,w0,Gamma )
L = Amp*Gamma.^2./((w-w0).^2+Gamma.^2);
end

