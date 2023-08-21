function [ FRF, FBB, comp ] = OMP( Fopt, Nrf, At )
K = size(Fopt,3); Ns = Nrf;
Nt = size(Fopt,1);
Nray = size(At,2);

comp = 0;
FRF = [];
Fres = Fopt;
for i = 1:Nrf
    temp = 0;
    for k = 1:K
        PU(:,:,k) = At' * Fres(:,:,k);
        comp = comp + count_flops(At',Fres(:,:,k));
        temp = temp + sum( abs(PU(:,:,k)).^2, 2 );
    end
    %temp
    [aa,bb] = max(temp);
    FRF = [FRF , sqrt(Nt) * At(:,bb)];
    for k = 1:K
        FBB{k} = pinv(FRF) * Fopt(:,:,k);
        comp = comp + Nt*Nrf^2 + count_flops(pinv(FRF),Fopt(:,:,k));
        Fres(:,:,k) = (Fopt(:,:,k) - FRF * FBB{k}) / norm(Fopt(:,:,k) - FRF * FBB{k},'fro');
        comp = comp + count_flops(FRF,FBB{k});
    end
end

for k = 1:K
    FBB{k} = sqrt(Ns) * FBB{k} / norm(FRF * FBB{k},'fro');
    % check power constraint and sub-connected constraint
    if abs(norm(FRF * FBB{k},'fro')^2 - Ns) > 1e-4 
        error('check power constraint !!!!!!!!!!!!')
    end
end

end