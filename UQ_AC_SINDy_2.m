
clear; close all; clc
addpath(genpath('.'));

polyorder = 2;         
lambda = 0.001;        
dt = 0.005;             
tau = 0.333;           
hill_n = 5;        
hill_P0 = 2.4;      
N = 1900;

load("C:\Users\nickj\MATLAB Drive\Sparse Dynamics SINDy\mRNA data\FH3mRNA_EXTRAP.mat");
load("C:\Users\nickj\MATLAB Drive\Sparse Dynamics SINDy\Hes1 data\FH3hes1_EXTRAP.mat");

load("C:\Users\nickj\MATLAB Drive\Sparse Dynamics SINDy\Hes1 data\hes1.mat")
load("C:\Users\nickj\MATLAB Drive\Sparse Dynamics SINDy\Hes1 data\hes1time.mat")
load("C:\Users\nickj\MATLAB Drive\Sparse Dynamics SINDy\mRNA data\mRNA.mat")
load("C:\Users\nickj\MATLAB Drive\Sparse Dynamics SINDy\mRNA data\mRNAtime.mat")

t = (0:dt:15)';
t_all = (0:dt:35)';
M_data_all = FH3mRNA_EXTRAP(:);
P_data_all = FH3hes1_EXTRAP(:);

M_data = FH3mRNA_EXTRAP(1:3001);
P_data = FH3hes1_EXTRAP(1:3001);


t_delay = t - tau;
P_tau = interp1(t, P_data, max(t_delay,0), 'linear');
HillDelay = 1 ./ (1 + (P_tau / hill_P0).^hill_n);  

Theta_poly = polyLib_MP(M_data, P_data, polyorder);
Theta = [Theta_poly, HillDelay];

sgolay_poly = 3; sgolay_frame = 11;
dMdt = sgolayfilt(gradient(M_data, dt), sgolay_poly, sgolay_frame);
dPdt = sgolayfilt(gradient(P_data, dt), sgolay_poly, sgolay_frame);


dMdt = dMdt(1:N);
dPdt = dPdt(1:N);
Theta = Theta(1:N,:);

% Normalize
colscale     = vecnorm(Theta,2,1); colscale(colscale==0) = 1;
ThetaN       = Theta ./ colscale;

targetScaleM = norm(dMdt, 2);
targetScaleP = norm(dPdt, 2);
dSdtN = [dMdt / targetScaleM, dPdt / targetScaleP];


% apex vals
polyscale = colscale(2:polyorder+1);
apex = 4.5;
powers = 1:polyorder;
apex_vals = (apex.^powers) ./ polyscale;

% Sparse regression
XiN = ThetaN\dSdtN;

ThetaN_ejected = ThetaN(:,1:(end-1));
n = 2;
eta = 0.8;
 
wM = ones(size(ThetaN_ejected, 2), 1);
wP = ones(size(ThetaN_ejected, 2), 1);
XiN_var = zeros(size(XiN)); 



c_dSdt    = [0.00 0.00 0.00];
c_ejected = [0.00 0.45 0.74];
c_hill    = [0.85 0.33 0.10];

for k = 1:12
    for ind = 1:n
       
        if ind == 1
            Hill_reduction = ThetaN(:, end) * XiN(end, ind);
            current_w = wM;
        else
            Hill_reduction = zeros(N,1);
            XiN(end, 2) = 0;
            current_w = wP;
        end
        
        dSdtN_ejected = dSdtN(:, ind) - Hill_reduction;
        XiN_notejected = XiN(1:end-1, ind);
        
        figure('Color','w','Units','normalized');
        plot(dSdtN(1:1900,ind),          '-', 'Color', c_dSdt,    'LineWidth', 1.0, 'DisplayName', 'dS/dt');        hold on;
        plot(dSdtN_ejected(1:1900),      '-', 'Color', c_ejected, 'LineWidth', 1.0, 'DisplayName', 'dS/dt Ejected');
        plot(Hill_reduction(1:1900),     '-', 'Color', c_hill,    'LineWidth', 1.0, 'DisplayName', 'Hill'); hold off;
       
        

        if ind == 1
            force = sum(XiN_notejected(2:polyorder+1) .* apex_vals');
            Hs_val = abs(force) * colscale(end);
            XiN(end, 1) = (1-eta)*XiN(end, 1) + eta*Hs_val;

            hill_var = sum(XiN_var(2:polyorder+1, 1) .* (apex_vals'.^2));
            XiN_var(end, 1) = hill_var * (colscale(end)^2);
        end

        ThetaN_weighted = ThetaN_ejected ./ current_w';
        
        biginds = current_w < 1e2; 
        if any(biginds)
           
           
            lb = -inf(sum(biginds), 1);
            ub =  inf(sum(biginds), 1);
            
           
            if ind == 1
                lb(1) = -0.1;
                ub(1) = 0.1;
                ub(2:3) = -1;
                lb(4:5) = 0;
                ub(4:5) = 0;
 
            end
             if ind == 2
                
                lb(2:3) = 0;
                ub(2:3) = 1;
                ub(4:5) = 0;
            end

            
            Xi_weighted = lsqlin(ThetaN_weighted(:, biginds), dSdtN_ejected, [], [], [], [], lb, ub);
            
          
            XiN_notejected(biginds) = Xi_weighted;
        end
        
      priorType = 'Laplace';
        eps_sparsity = 1e-3;
        switch priorType
            case 'Laplace'
                new_w = 1 ./ (abs(XiN_notejected) + eps_sparsity);
                
            case 'SpikeSlab'
                v0 = 1e-4; 
                v1 = 1.0; 
              
                pi_incl = 1 ./ (1 + (v1/v0)*exp(-XiN_notejected.^2 / (2*v0)));
                new_w = 1 ./ (pi_incl*v1 + (1-pi_incl)*v0);
                
            case 'Horseshoe'
                tau0 = 0.05; 
                
                new_w = 1 ./ (tau0^2 * (XiN_notejected.^2 + eps_sparsity));
        end
        
     
        if ind == 1
            wM = (0.7 * wM) + (0.3 * new_w); 
            current_w = wM;
        else
            wP = (0.7 * wP) + (0.3 * new_w);
            current_w = wP;
        end
      
        
        XiN(1:end-1, ind) = XiN_notejected;
        
        resid_var = var(dSdtN_ejected - ThetaN_ejected * XiN_notejected);
        H = (ThetaN_ejected' * ThetaN_ejected) / resid_var + diag(current_w);
        XiN_var(1:end-1, ind) = diag(inv(H));
    end
end

clc

% Explicitly denormalize each column
Xi = zeros(size(XiN));
Xi(:,1) = (XiN(:,1) * targetScaleM) ./ colscale';
Xi(:,2) = (XiN(:,2) * targetScaleP) ./ colscale';


% Extract for the ODE loop
XiM = Xi(:, 1);
XiP = Xi(:, 2);
disp(XiM)
disp(XiP)
termNames = [polyLibNames_Monly('M', polyorder), {'Hill(P_\tau)'}];

    a_offset = Xi(end,1);  
    b_offset = -Xi(end,1);       
    t_offset_start = 0; 
    t_offset_end   = tau; 
    offset_fun = @(tt) ((tt >= t_offset_start & tt <= t_offset_end) .* (a_offset*tt + b_offset));

% Forward integration
M_id = zeros(size(M_data));
M_id(1) = M_data(1);
P_id = zeros(size(P_data));
P_id(1) = P_data(1);

for k = 2:length(t_all)
    M_prev = M_id(k-1);
    P_prev = P_id(k-1);
    if k <= round(tau/dt)+1
        P_tau_k = interp1(t, P_data, max(t(k)-tau, 0), 'linear');
    else
        P_tau_k = P_id(k - round(tau/dt));
    end
    Hill_k  = 1 / (1 + (P_tau_k/hill_P0)^hill_n);
    row_polyM = polyRow_MP(M_prev, P_prev, polyorder);
    rowM = [row_polyM, Hill_k];
    dMdt_k = rowM * XiM + offset_fun(t_all(k)); 
    row_polyP = polyRow_MP(M_prev, P_prev, polyorder);
    rowP = [row_polyP, Hill_k];
    dPdt_k = rowP * XiP ; 
    M_id(k) = M_prev + dt * dMdt_k;
    P_id(k) = P_prev + dt * dPdt_k;

end

  % Helper functions
    
   function Theta = polyLib_MP(M, P, n)
    Theta = ones(size(M));

    for k = 1:n
        Theta = [Theta, M.^k];
    end

    for k = 1:n
        Theta = [Theta, P.^k];
    end
end
    
    function row = polyRow_MP(M, P, n)
    row = 1;

    for k = 1:n
        row = [row, M^k];
    end

    for k = 1:n
        row = [row, P^k];
    end
end
   








%% Monte Carlo Uncertainty Sampling
num_samples = 100; 
t_all = (0:dt:35)';
M_samples = zeros(length(t_all), num_samples);
P_samples = zeros(length(t_all), num_samples);
% Denormalize mRNA equation variances (column 1)
 Xi_var_phys = zeros(size(XiN_var));
Xi_var_phys(:, 1) = XiN_var(:, 1) .* (targetScaleM ./ colscale').^2;

% Denormalize Protein equation variances (column 2)
Xi_var_phys(:, 2) = XiN_var(:, 2) .* (targetScaleP ./ colscale').^2;
for s = 1:num_samples
    % Draw coefficients from the physical Gaussian distributions
    % Standard Deviation = sqrt(Variance)
    XiM_sample = XiM + sqrt(Xi_var_phys(:,1)) .* randn(size(XiM));
    XiP_sample = XiP + sqrt(Xi_var_phys(:,2)) .* randn(size(XiP));
    
    % Ensure physical viability (Hill coefficient must be positive)
    XiM_sample(end) = max(1e-6, XiM_sample(end));
    % Temporary storage for this specific sample's trajectory
    M_temp = zeros(size(t_all)); P_temp = zeros(size(t_all));
    M_temp(1) = M_data(1); P_temp(1) = P_data(1);
    
    % Integration Loop (using the sampled coefficients)
    for k = 2:length(t_all)
        M_prev = M_temp(k-1);
        P_prev = P_temp(k-1);
        
        % Handle Delay
        if k <= round(tau/dt)+1
            P_tau_k = interp1(t, P_data, max(t(k)-tau, 0), 'linear');
        else
            P_tau_k = P_temp(k - round(tau/dt));
        end
        
        Hill_k  = 1 / (1 + (P_tau_k/hill_P0)^hill_n);
        row_poly = polyRow_MP(M_prev, P_prev, polyorder);
        
        % Calculate derivatives with sampled coefficients
        dMdt_s = [row_poly, Hill_k] * XiM_sample + offset_fun(t_all(k));
        dPdt_s = [row_poly, Hill_k] * XiP_sample;
        
        % Euler Step (ensure values stay positive)
        M_temp(k) = max(0, M_prev + dt * dMdt_s);
        P_temp(k) = max(0, P_prev + dt * dPdt_s);
    end
    
    M_samples(:, s) = M_temp;
    P_samples(:, s) = P_temp;
end

%% Plotting the UQ Results
% Calculate mean and standard deviation across samples
M_med = median(M_samples, 2);
M_low = quantile(M_samples, 0.05, 2);
M_high = quantile(M_samples, 0.95, 2);

P_med = median(P_samples, 2);
P_low = quantile(P_samples, 0.05, 2);
P_high = quantile(P_samples, 0.95, 2);figure('Color', 'k');

% --- mRNA Plot ---
subplot(2,1,1);
% 90% Confidence Interval (5th to 95th percentile)
fill([t_all; flipud(t_all)], [M_high; flipud(M_low)], ...
    [0 0.6 0.6], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', '90% CI'); 
hold on;
plot(t_all, M_med, 'c-', 'LineWidth', 2, 'DisplayName', 'Median Model');
plot(t_all, M_data_all, 'w--', 'LineWidth', 1, 'DisplayName', 'True Data');
title('mRNA (M) - Median Trajectory & Quantile Envelope', 'Color', 'w');
grid on; set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');
ylabel('Concentration'); legend('TextColor', 'w');
plot(mRNAtime, mRNA, 'wo')

% --- Protein Plot ---
subplot(2,1,2);
fill([t_all; flipud(t_all)], [P_high; flipud(P_low)], ...
    [0.6 0 0.6], 'FaceAlpha', 0.3, 'EdgeColor', 'none'); 
hold on;
plot(t_all, P_med, 'm-', 'LineWidth', 2);
plot(t_all, P_data_all, 'w--', 'LineWidth', 1);
title('Protein (P) - Median Trajectory & Quantile Envelope', 'Color', 'w');
grid on; set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');
ylabel('Concentration'); xlabel('Time (h)');
plot(hes1time, hes1, 'wo')

%% 1. Prepare Data for Plotting
% Use physical space coefficients for meaningful interpretation
Xi_phys = Xi; % From your denormalization step
% Denormalize variance: Var_phys = Var_norm * (TargetScale/Colscale)^2
Xi_var_phys = XiN_var .* ([targetScaleM, targetScaleP] ./ colscale').^2;

% Define term names for the legend/labels
% Adjust this list to match your polyLib_MP order
names = {'Constant', 'M', 'M^2', 'P', 'P^2', 'Hill(P)'};

figure('Color', 'k', 'Name', 'Posterior Distributions of Identified Terms');

colors = lines(length(names));

% Create subplots for mRNA (M) and Protein (P) equations
for ind = 1:2
    subplot(2, 1, ind);
    hold on;
    
    % Find indices where the coefficient is not zero
    val_idx = find(Xi_phys(:, ind) < 10^6);
    
    for j = 1:length(val_idx)
        idx = val_idx(j);
        mu = Xi_phys(idx, ind);
        sigma = sqrt(Xi_var_phys(idx, ind));
        
        % Generate a range for the x-axis (3 standard deviations)
        x_range = linspace(mu - 4*sigma, mu + 4*sigma, 200);
        % Calculate Gaussian PDF
        y_pdf = (1/(sigma * sqrt(2*pi))) * exp(-0.5 * ((x_range - mu)/sigma).^2);
        
        % Plot the distribution
        plot(x_range, y_pdf, 'LineWidth', 2, 'Color', colors(idx,:), ...
            'DisplayName', names{idx});
        
        % Add a shaded area under the curve
        fill([x_range, fliplr(x_range)], [y_pdf, zeros(size(y_pdf))], ...
            colors(idx,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
    
    title(['Equation ' num2str(ind) ' ( ' (char(76+ind)) ' ) term distributions']);
    xlabel('Coefficient Value');
    ylabel('Probability Density');
    legend('Location', 'northeastoutside');
    grid on;
    ylim([0 5]);
   
  
end
figure('Color','k');

subplot(2,1,1);
plot(t, M_data,'w-','LineWidth',1.2); hold on;
plot(t, M_id(1:3001),'c--','LineWidth',1.2);
xline((N*dt), ':', 'Color', [0.8 0.8 0.8], 'LineWidth',1.2);
title('mRNA (M)','Color','w'); grid on;
set(gca,'Color','k','XColor','w','YColor','w');
plot(mRNAtime, mRNA, 'wo')

subplot(2,1,2);
plot(t, P_data,'w-','LineWidth',1.2); hold on;
plot(t, P_id(1:3001),'m--','LineWidth',1.2);
xline((N*dt), ':', 'Color', [0.8 0.8 0.8], 'LineWidth',1.2);
title('Protein (P)','Color','w'); grid on;
xlabel('Time (h)','Color','w');
set(gca,'Color','k','XColor','w','YColor','w');

plot(hes1time, hes1, 'wo')





