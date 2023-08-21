function [FRF, FBB, comp] = AO(SNR,noisevar,Nrf,Frf0,H,Fopt)
comp = 0;
[Nt, Ns, K] = size(Fopt);
Hk = H(:,:,K/2);

%% Obtain analog precoder
F1 = Hk'*Hk;
gamma = SNR/(Nt*Nrf);
FRF = Frf0;
stopconverge = 0;
while (stopconverge == 0)
    Vrf_old = FRF;
    for jj=1:Nrf
        Vbar = FRF; Vbar(:,jj)=[];
        C = eye(Nrf-1)+(gamma/noisevar)*Vbar'*F1*Vbar;
        comp = comp + count_flops(Vbar',F1) + count_flops(Vbar'*F1,Vbar) + size(Vbar'*F1*Vbar,1)*size(Vbar'*F1*Vbar,2);
        
        G = (gamma/noisevar)*F1 - (gamma/noisevar)^2*F1*Vbar*(C^-1)*Vbar'*F1;
        comp = comp + Nrf^3 + count_flops(F1*Vbar,(C^-1)) + count_flops(F1*Vbar*(C^-1),Vbar'*F1) + 2*size(F1,1)*size(F1,2);
        for ii = 1:Nt
            gvec = G(ii,:);gvec(ii)=[];
            vvec = FRF(:,jj); vvec(ii)=[];
            eta = gvec*vvec;
            comp = comp + 2*Nt + 1;
            if(eta==0)
                FRF(ii,jj)=1;
            else
                FRF(ii,jj) = eta/abs(eta);
            end
        end
    end
    
    Frf_change = FRF - Vrf_old;
    if sum(abs(Frf_change(:))) < 0.001
        stopconverge = 1;
    end
end

%% digtal precoding
FBB = zeros(Nrf, Ns, K);
for kk = 1:K
    Heff = H(:,:,kk)*FRF;
    comp = comp + count_flops(H(:,:,kk),FRF);
    
    Q = FRF'*FRF;
    comp = comp + count_flops(FRF',FRF);
    [Us, Ss, Vs] = svd(Heff*Q^(-0.5),'econ');
    comp = comp + Nrf^3;
    
    Fbbtmp = Q^(-0.5)*Vs(:,1:Ns);
    comp = comp + count_flops(Q,Vs(:,1:Ns));
    
    FBB(:,:,kk) = sqrt(Ns) * Fbbtmp / norm(FRF * Fbbtmp,'fro'); % normalization
    comp = comp + count_flops(FRF,Fbbtmp);
end




