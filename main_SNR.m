clear all,clc
close all
warning off

addpath(pwd);
cd manopt;
addpath(genpath(pwd));
cd ..;

%% system parameters
Nt = 128;
Nr = 2;
Ns = 2;
Nrf = Ns;
L = log2(Nt);
Inet = 10;
K = 128;

%% simulation parameters
n_trials = 20;
sub_train = 1;
SNR_dB = -10:5:20;
SNR = 10.^(SNR_dB./10);
smax = length(SNR);% enable the parallel

%% compared schemes
run_MO = 1; run_SDR = 0; run_OMP = 1; run_AO = 1; run_DNN = 1;

%% load data
[H0,Fopt0,At0,f_DNN_all] = load_data(Nt,Nr,Nrf,K,L,'FC-HBF');

%% start simulations
R_opt = zeros(smax,n_trials);
R_MO = zeros(smax,n_trials);
R_OMP = zeros(smax,n_trials);
R_AO = zeros(smax,n_trials);
R_DNN0 = zeros(smax,n_trials);
R_DNN = zeros(smax,n_trials);
R_DNN_dyn_ES = zeros(smax,n_trials);
R_SDR = zeros(smax,n_trials);
R_DNN_fix = zeros(smax,n_trials);
R_DNN_dyn_BestHk = zeros(smax,n_trials);
R_subManNet = zeros(smax,n_trials);

for ii = 1:n_trials
    ii
    
    %% Channels
    H = H0(:,:,:,ii);
    Hc = H(:,:,end);
    Fopt = Fopt0(:,:,:,ii);
    At = At0(:,:,K,ii);
    
    %% OMP
    if run_OMP == 1
        [Frf_OMP, Fbb_OMP] = OMP(Fopt, Nrf, At);
    end
    
    %% Manifold solution
    if run_MO == 1
        Frf0 = exp(1i*unifrnd(0,2*pi,Nt,Nrf));
        [Frf_MO, Fbb_MO] = MO_AltMin(Fopt, Frf0);
    end
    
    %% SDR
    if run_SDR == 1
        [Frf_SDR, Fbb_SDR] = SDR_AltMin(Fopt,Nrf);
    end
    
    
    %% Evaluation
    for s = 1:smax
        R_opt(s,ii) = compute_rate(H,eye(Ns),Fopt,SNR(s),Ns,K,'opt');
        
        if run_AO == 1
            [Frf_AO, Fbb_AO] = AO(SNR(s),1,Nrf,ones(Nt,Nrf),H,Fopt);
            R_AO(s,ii) = compute_rate(H,Frf_AO,Fbb_AO,SNR(s),Ns,K,'AO');
        end
        
        if run_MO == 1
            R_MO(s,ii) = compute_rate(H,Frf_MO,Fbb_MO,SNR(s),Ns,K,'MO');
        end
        
        if run_OMP == 1
            R_OMP(s,ii) = compute_rate(H,Frf_OMP,Fbb_OMP,SNR(s),Ns,K,'OMP');
        end
        
        if run_SDR == 1
            R_SDR(s,ii) = compute_rate(H,Frf_SDR,Fbb_SDR,SNR(s),Ns,K,'SDR');
        end
        
        %% DNN
        if run_DNN == 1
            
            %% non-iterative, FC-HBF
            [Frf_DNN0, Fbb_DNN0] = ManNet(H,f_DNN_all(1,ii,:),Nt,Nrf,Ns,K,Fopt);
            R_DNN0(s,ii) = compute_rate(H,Frf_DNN0,Fbb_DNN0,SNR(s),Ns,K,'ManNet');
            
            %% iterative, FC-HBF
            [Frf_DNN, Fbb_DNN] = ManNet(H,f_DNN_all(Inet,ii,:),Nt,Nrf,Ns,K,Fopt);
            R_DNN(s,ii) = compute_rate(H,Frf_DNN,Fbb_DNN,SNR(s),Ns,K,'ManNet');
            
            %% iterative, Fixed FC-HBF
            [Frf_DNN_fix, Fbb_DNN_fix] = ManNet_sub(H,f_DNN_all(Inet,ii,:),Nt,Nrf,Ns,K,Fopt,SNR(s),0);
            R_DNN_fix(s,ii) = compute_rate(H,Frf_DNN_fix,Fbb_DNN_fix,SNR(s),Ns,K,'ManNet');

            %% dynamic SC-HBF, best Hk
            [Frf_DNN_dyn_BestH, Fbb_DNN_dyn_BestH] = ManNet_sub(H,f_DNN_all(Inet,ii,:),Nt,Nrf,Ns,K,Fopt,SNR(s),1);
            R_DNN_dyn_BestHk(s,ii) = compute_rate(H,Frf_DNN_dyn_BestH,Fbb_DNN_dyn_BestH,SNR(s),Ns,K,'ManNet');

            %% dynamic SC-HBF, heuristic
            [Frf_DNN_dyn_ES, Fbb_DNN_dyn_ES] = ManNet_sub(H,f_DNN_all(Inet,ii,:),Nt,Nrf,Ns,K,Fopt,SNR(s),2);
            R_DNN_dyn_ES(s,ii) = compute_rate(H,Frf_DNN_dyn_ES,Fbb_DNN_dyn_ES,SNR(s),Ns,K,'ManNet');
        end
    end
    
end


%% Plot figures
%load("rate_SNR_128")
% load("rate_SNR_128_SDR_AltMin")
figure
plot(SNR_dB,mean(R_opt,2),'--b','LineWidth',1.5, 'MarkerSize',7); hold on;
plot(SNR_dB,mean(R_MO,2),':ko','LineWidth',1.5, 'MarkerSize',7); hold on;
% plot(SNR_dB,mean(R_PE,2),'-g>','LineWidth',1.5); hold on;
plot(SNR_dB,mean(R_AO,2),'-cs','LineWidth',1.5, 'MarkerSize',7); hold on;
plot(SNR_dB,mean(R_OMP,2),'-kd','LineWidth',1.5, 'MarkerSize',7); hold on;
plot(SNR_dB,mean(R_DNN0,2),'--b+','LineWidth',1.5, 'MarkerSize',7); hold on;
plot(SNR_dB,mean(R_DNN,2),'-rp','LineWidth',1.5, 'MarkerSize',7); hold on;


legend('FC-HBF, Optimal DBF',...
    'FC-HBF, MO-AltMin',...
    'FC-HBF, AO',...
    'FC-HBF, OMP',...
    'FC-HBF, ManNet, $\mathcal{I}_{\mathrm{net}} = 1$',...
    'FC-HBF, ManNet, $\mathcal{I}_{\mathrm{net}} = 10$',...
    'Location','Best','fontsize',10,'interpreter','latex')
xlabel('SNR [dB]','fontsize',12,'interpreter','latex');
ylabel('Spectral efficiency [bits/s/Hz]','fontsize',12,'interpreter','latex');
xticks(SNR_dB)

grid on
hold on

%% SC-HBF trained ---------------------------------------------------------------------------------
if sub_train == 1
    %% load data
    [H0,Fopt0,At0,f_DNN_all] = load_data(Nt,Nr,Nrf,K,L,'SC-HBF');
    
    smax = length(SNR);% enable the parallel
    for ii = 1:n_trials
        ii
        
        %% Channels
        H = H0(:,:,:,ii);
        Hc = H(:,:,end);
        Fopt = Fopt0(:,:,:,ii);
        At = At0(:,:,K,ii);
        
        %% Evaluation
        for s = 1:smax
            Rmax = 0;
            for iter = 1:Inet
                
                %% Dynamic SC-HBF with subManNet
                [Frf_DNN, Fbb_DNN] = ManNet_sub_train(H,f_DNN_all(iter,ii,:),Nt,Nrf,Ns,K,Fopt,SNR(s));
                Rtmp = compute_rate(H,Frf_DNN,Fbb_DNN,SNR(s),Ns,K,'ManNet');
                if Rtmp > Rmax
                    Rmax = Rtmp;
                end
            end
            R_subManNet(s,ii) = Rmax;
        end
    end
    figure
    plot(SNR_dB,mean(R_MO,2),':ko','LineWidth',1.5, 'MarkerSize',7); hold on;
    plot(SNR_dB,mean(R_subManNet,2),'-mx','LineWidth',1.5, 'MarkerSize',7); hold on;
    plot(SNR_dB,mean(R_DNN_fix,2),'--^','Color',[0 0.4470 0.7410],'LineWidth',1.5, 'MarkerSize',7); hold on;
    plot(SNR_dB,mean(R_DNN_dyn_ES,2),'-b*','LineWidth',1.5, 'MarkerSize',7); hold on;
    plot(SNR_dB,mean(R_DNN_dyn_BestHk,2),'-r+','LineWidth',1.5, 'MarkerSize',7); hold on;
    % plot(SNR_dB,mean(R_SDR,2),'-.k*','LineWidth',1.5); hold on;
    
    legend('FC-HBF, MO-AltMin',...
        'Dynamic SC-HBF, subManNet',...
        'Fixed SC-HBF, ManNet',...
        'Dynamic SC-HBF, ManNet, heuristic',...
        'Dynamic SC-HBF, ManNet, with $\mathbf{H}[k^{\star}]$',...
        'Location','Best','fontsize',10,'interpreter','latex')
    xlabel('SNR [dB]','fontsize',12,'interpreter','latex');
    ylabel('Spectral efficiency [bits/s/Hz]','fontsize',12,'interpreter','latex');
    xticks(SNR_dB)
    grid on
    hold on
end

%% save results
% save("rate_SNR_128_SDR_AltMin")