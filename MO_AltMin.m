function [FRF,FBB, comp] = MO_AltMin(Fopt, FRF)
comp = 0;
[Nt, Ns, K] = size(Fopt);
Nrf = size(FRF,2);
FBB = zeros(Nrf, Ns, K);
if Nt > Nrf % HBF
    y = [];
    %FRF = exp(1i*unifrnd(0,2*pi,Nt,NRF));
    while (isempty(y) || abs(y(1)-y(2)) > 1e-3)
        y = [0,0];
        for k = 1:K
            FBB(:,:,k) = pinv(FRF) * Fopt(:,:,k);
            comp = comp + Nt*Nrf^2 + count_flops(pinv(FRF),Fopt(:,:,k));
            y(1) = y(1) + norm(Fopt(:,:,k) - FRF * FBB(:,:,k),'fro')^2;
            comp = comp + count_flops(FRF,FBB(:,:,k)) + Nrf*Ns;
        end
        [FRF, y(2), comp_in] = sig_manif(Fopt, FRF, FBB);
        comp = comp + comp_in;
        %abs(y(1)-y(2))
    end
else
    FRF = eye(Nrf);
    FBB = Fopt;
end

for k = 1:K
    FBB(:,:,k) = sqrt(Ns) * FBB(:,:,k) / norm(FRF * FBB(:,:,k),'fro');
    if abs(norm(FRF * FBB(:,:,k),'fro')^2 - Ns) > 1e-4
        error('check power constraint !!!!!!!!!!!!')
    end
end

end