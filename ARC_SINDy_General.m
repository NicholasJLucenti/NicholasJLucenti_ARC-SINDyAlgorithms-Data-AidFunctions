clear; close all;  clc
addpath(genpath('.'));

% ARC-SINDy Hyperparameters
polyorder    = 2;         
dt           = 0.005;             
tau          = 0.333;           
N            = 1900;         
eta          = 0.6;          


hill_n       = 5;        
hill_k0      = 2.4;      
hill_func    = @(p, k, n) 1 ./ (1 + (p./k).^n);

load("FH3mRNA_EXTRAP.mat"); % Load x_data_source
load("FH3hes1_EXTRAP.mat"); % Load y_data_source

load("C:Hes1 data\hes1.mat")
load("C:Hes1 data\hes1time.mat")
load("C:mRNA data\mRNA.mat")
load("C:mRNA data\mRNAtime.mat")


t     = (0:dt:15)';
t_all = (0:dt:35)';

x_data_all   = FH3mRNA_EXTRAP(:);
y_data_all   = FH3hes1_EXTRAP(:);
x_data       = x_data_all(1:3001);
y_data       = y_data_all(1:3001);

t_delay = t - tau;
y_tau   = interp1(t, y_data, max(t_delay,0), 'linear');

% Generate Library
HillDelay = hill_func(y_tau, hill_k0, hill_n);  
Theta_poly = polyLib_gen(x_data, y_data, polyorder);
Theta = [Theta_poly, HillDelay];

% Numerical Derivatives
sgolay_p = 3; sgolay_f = 11;
dxdt = sgolayfilt(gradient(x_data, dt), sgolay_p, sgolay_f);
dydt = sgolayfilt(gradient(y_data, dt), sgolay_p, sgolay_f);

% Truncate to Training Window
dxdt = dxdt(1:N);
dydt = dydt(1:N);
Theta = Theta(1:N,:);

colscale      = vecnorm(Theta,2,1); colscale(colscale==0) = 1;
ThetaN        = Theta ./ colscale;
targetScaleX  = norm(dxdt, 2);
targetScaleY  = norm(dydt, 2);
dSdtN         = [dxdt / targetScaleX, dydt / targetScaleY];

n = 2;
XiN = ThetaN \ dSdtN; % Initial LS

ThetaN_ejected = ThetaN(:,1:(end-1));


wx = ones(size(ThetaN_ejected, 2), 1);
wy = ones(size(ThetaN_ejected, 2), 1);
XiN_var = zeros(size(XiN)); 

for k = 1:10
    poly_indices = 2:polyorder+1;
    

    for ind = 1:n
        if ind == 1

            xi_poly  = XiN(poly_indices, ind);
            var_poly = XiN_var(poly_indices, ind);
           
            [Hs_val, Hs_var] = calculate_Hyperstate_Hes1(xi_poly, var_poly, colscale, targetScaleX);
            
            % Update Hill term in the normalized matrix
            XiN(end, 1) = (1-eta)*XiN(end, 1) + eta*Hs_val;
            XiN_var(end, 1) = Hs_var / (colscale(end)^2); % Final scaling for UQ
            
            Hill_reduction = ThetaN(:, end) * XiN(end, ind);
            current_w = wx;
            
        elseif ind == 2
            Hill_reduction = zeros(N,1);
            XiN(end, 2) = 0;
            current_w = wy;
        end
        
        dSdtN_ejected = dSdtN(:, ind) - Hill_reduction;
        XiN_notejected = XiN(1:end-1, ind);
        
    
        
        
        ThetaN_weighted = ThetaN_ejected ./ current_w';
        biginds = current_w < 1e2; 
        
        if any(biginds)
            
            if ind == 1 
                lb(1)   = -0.1;   ub(1) = 0.1; 

                lb(2:3) = -inf;   ub(2:3) = -1; 

                lb(4:5) = 0;      ub(4:5) = 0; 

            elseif ind == 2 
                lb(1)   = -inf;   ub(1) = inf; 

                lb(2:3) = 1;      ub(2:3) = inf; 

                lb(4:5) = -inf;   ub(4:5) = 0;   

            end
            
            XiN_notejected(biginds) = lsqlin(ThetaN_weighted(:, biginds), dSdtN_ejected, [], [], [], [], lb, ub);
        end
        

        priorType = 'Horseshoe';  eps_sparsity = 1e-3;
        switch priorType
            case 'Laplace'
                new_w = 1 ./ (abs(XiN_notejected) + eps_sparsity);
                
            case 'SpikeSlab'
                v0 = 1e-4;  v1 = 1.0; 
                pi_incl = 1 ./ (1 + (v1/v0)*exp(-XiN_notejected.^2 / (2*v0)));
                new_w = 1 ./ (pi_incl*v1 + (1-pi_incl)*v0);
                
            case 'Horseshoe'
                tau0 = 0.05; 
                new_w = 1 ./ (tau0^2 * (XiN_notejected.^2 + eps_sparsity));
        end

        if ind == 1
            wx = (eta * wx) + ((1-eta) * new_w); current_w = wx;
        elseif ind == 2
            wy = (eta * wy) + ((1-eta) * new_w); current_w = wy;
        end
      
        XiN(1:end-1, ind) = XiN_notejected;
        
        % Bayesian Variance Update
        resid_var = var(dSdtN_ejected - ThetaN_ejected * XiN_notejected);
        H = (ThetaN_ejected' * ThetaN_ejected) / resid_var + diag(current_w);
        XiN_var(1:end-1, ind) = diag(inv(H));
    end
end

Xi = zeros(size(XiN));
Xi(:,1) = (XiN(:,1) * targetScaleX) ./ colscale';
Xi(:,2) = (XiN(:,2) * targetScaleY) ./ colscale';

% FORWARD INTEGRATION 
XiX = Xi(:, 1); XiY = Xi(:, 2);
x_id = zeros(size(x_data_all)); x_id(1) = x_data(1);
y_id = zeros(size(y_data_all)); y_id(1) = y_data(1);

offset_val = Xi(end,1); 
offset_fun = @(tt) ((tt <= tau) .* (offset_val*tt - offset_val));

for k = 2:length(t_all)
    x_prev = x_id(k-1); y_prev = y_id(k-1);
    
    if k <= round(tau/dt)+1
        y_tau_k = interp1(t, y_data, max(t_all(k)-tau, 0), 'linear');
    else
        y_tau_k = y_id(k - round(tau/dt));
    end
    
    H_k = hill_func(y_tau_k, hill_k0, hill_n);
    
    phi = [polyRow_gen(x_prev, y_prev, polyorder), H_k];
    
    x_id(k) = x_prev + dt * (phi * XiX + offset_fun(t_all(k)));
    y_id(k) = y_prev + dt * (phi * XiY);
end

%% HELPER FUNCTIONS 
function Theta = polyLib_gen(x, y, n)
    Theta = ones(size(x));
    for k = 1:n, Theta = [Theta, x.^k]; end
    for k = 1:n, Theta = [Theta, y.^k]; end
end

function row = polyRow_gen(x, y, n)
    row = 1;
    for k = 1:n, row = [row, x^k]; end
    for k = 1:n, row = [row, y^k]; end
end
   
figure('Color','k');
plot(t_all, x_data_all,'w-','LineWidth',1.2); hold on;
plot(t_all, x_id,'c--','LineWidth',1.2);
xline((N*dt), ':', 'Color', [0.8 0.8 0.8], 'LineWidth',1.2);
title('mRNA (M)','Color','w'); grid on;
set(gca,'Color','k','XColor','w','YColor','w');
xlim([0, 10]);
hold off;

figure('Color','k');
plot(t_all, y_data_all,'w-','LineWidth',1.2); hold on;
plot(t_all, y_id,'m--','LineWidth',1.2);
xline((N*dt), ':', 'Color', [0.8 0.8 0.8], 'LineWidth',1.2);
title('Protein (P)','Color','w'); grid on;
xlabel('Time (h)','Color','w');
xlim([0, 10]);
clc
disp(Xi)









%% =====================================================



Xi_var_phys = XiN_var .* ([targetScaleX, targetScaleY] ./ colscale').^2;

n_samples = 200;   % number of Monte Carlo draws
nT        = length(t_all);
X_mc      = zeros(nT, n_samples);
Y_mc      = zeros(nT, n_samples);

rng(42);  

for s = 1:n_samples

    Xi_s = zeros(size(Xi));
    for col = 1:2
        for row = 1:size(Xi,1)
            Xi_s(row,col) = Xi(row,col) + ...
                sqrt(Xi_var_phys(row,col)) * randn();
        end
    end

    XiX_s = Xi_s(:,1);
    XiY_s = Xi_s(:,2);

    % forward integration loop
    x_s = zeros(nT,1);  x_s(1) = x_data(1);
    y_s = zeros(nT,1);  y_s(1) = y_data(1);

    offset_val_s = Xi_s(end,1);
    offset_fun_s = @(tt) ((tt <= tau) .* (offset_val_s*tt - offset_val_s));

    for k = 2:nT
        x_prev = x_s(k-1);
        y_prev = y_s(k-1);

        if k <= round(tau/dt)+1
            y_tau_k = interp1(t, y_data, max(t_all(k)-tau,0),'linear');
        else
            y_tau_k = y_s(k - round(tau/dt));
        end

        H_k = hill_func(y_tau_k, hill_k0, hill_n);
        phi = [polyRow_gen(x_prev, y_prev, polyorder), H_k];

        x_s(k) = x_prev + dt*(phi*XiX_s + offset_fun_s(t_all(k)));
        y_s(k) = y_prev + dt*(phi*XiY_s);

        x_s(k) = max(-20, min(x_s(k), 20));
        y_s(k) = max(-20, min(y_s(k), 20));
    end

    X_mc(:,s) = x_s;
    Y_mc(:,s) = y_s;
end

prct  = [5, 95];
X_env = prctile(X_mc, prct, 2);
Y_env = prctile(Y_mc, prct, 2);

X_low = X_env(:,1);  X_high = X_env(:,2);
Y_low = Y_env(:,1);  Y_high = Y_env(:,2);

X_med = median(X_mc, 2);
Y_med = median(Y_mc, 2);

figure('Color','w','Name','UQ-SINDy Monte Carlo Envelopes');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

axStyle = @(ax) set(ax, ...
    'Color','w','XColor','k','YColor','k', ...
    'FontSize',9,'LineWidth',1,'TickDir','out');

x_lim_mc = [0, 12];
y_lim_M   = [-0.5,  7];
y_lim_P   = [-0.5, 13];

% --- mRNA ---
ax1 = nexttile;
fill([0, N*dt, N*dt, 0], [y_lim_M(1), y_lim_M(1), y_lim_M(2), y_lim_M(2)], ...
    [0.85 0.85 0.85],'FaceAlpha',0.4,'EdgeColor','none','HandleVisibility','off');
hold on;
fill([t_all; flipud(t_all)], [X_high; flipud(X_low)], ...
    [0 0.45 0.7],'FaceAlpha',0.25,'EdgeColor','none','DisplayName','90% CI');
plot(t_all, X_med,      'b-',  'LineWidth',1.8, 'DisplayName','Median Model');
plot(t_all, x_data_all, 'k-',  'LineWidth',1.0, 'DisplayName','Sinusoidal Regression');
plot(mRNAtime, mRNA, 'ko','LineWidth',0.5, 'DisplayName','Observed Data')
title('Hes1 mRNA — Monte Carlo UQ Propagation ()','FontSize',9,'Color','k');
ylabel('Hes1 mRNA Concentration (\mug/\muL)','FontSize',9);
leg1 = legend('Location','northeast','FontSize',9,'Box','on');
set(leg1,'TextColor','k','Color','w','EdgeColor','k');
xlim(ax1, x_lim_mc); ylim(ax1, y_lim_M);
axStyle(ax1); box(ax1,'off'); grid(ax1,'off');
hold off;

% --- Protein ---
ax2 = nexttile;
fill([0, N*dt, N*dt, 0], [y_lim_P(1), y_lim_P(1), y_lim_P(2), y_lim_P(2)], ...
    [0.85 0.85 0.85],'FaceAlpha',0.4,'EdgeColor','none','HandleVisibility','off');
hold on;
fill([t_all; flipud(t_all)], [Y_high; flipud(Y_low)], ...
    [0.8 0.1 0.1],'FaceAlpha',0.25,'EdgeColor','none','DisplayName','90% CI');
plot(t_all, Y_med,      'r-',  'LineWidth',1.8, 'DisplayName','Median Model');
plot(t_all, y_data_all, 'k-',  'LineWidth',1.0, 'DisplayName','Sinusoidal Regression');
plot(hes1time, hes1, 'ko','LineWidth',0.5, 'DisplayName','Observed Data')
title('Hes1 Protein — Monte Carlo UQ Propagation ()','FontSize',9,'Color','k');
xlabel('Time (h)','FontSize',9);
ylabel('Hes1 Protein Concentration (\mug/\muL)','FontSize',9);
leg2 = legend('Location','northeast','FontSize',9,'Box','on');
set(leg2,'TextColor','k','Color','w','EdgeColor','k');
xlim(ax2, x_lim_mc); ylim(ax2, y_lim_P);
axStyle(ax2); box(ax2,'off'); grid(ax2,'off');
hold off;

% Bayesian Posterior Distributions
Xi_phys = Xi;
names = {'Constant','M','M^2','P','P^2','Hill(P)'};

figure('Color','w','Name','Posterior Distributions of Identified Terms');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

colors1 = lines(length(names)) * 0.65;
eqTitles = {'Hes1 mRNA','Hes1 Protein'};
eqLabels  = {'M','P'};

x_lim_pdf = {[-2.1,  17],  [-5,  3]};   % {mRNA, Protein}
y_lim_pdf = {[ 0, 10],  [ 0, 10]};

for ind = 1:2

    ax = nexttile;
    hold on;

    val_idx = find(Xi_phys(:,ind) < 1e6);

    for j = 1:length(val_idx)
        idx   = val_idx(j);
        mu    = Xi_phys(idx, ind);
        sigma = sqrt(Xi_var_phys(idx, ind));

        x_range = linspace(mu - 4*sigma, mu + 4*sigma, 500);
        y_pdf   = (1/(sigma*sqrt(2*pi))) * exp(-0.5*((x_range-mu)/sigma).^2);

        plot(x_range, y_pdf, ...
            'LineWidth',1.8,'Color',colors1(idx,:),'DisplayName',names{idx});

        fill([x_range, fliplr(x_range)], [y_pdf, zeros(size(y_pdf))], ...
            colors1(idx,:),'FaceAlpha',0.15,'EdgeColor','none', ...
            'HandleVisibility','off');
    end

    xlim(ax, x_lim_pdf{ind});
    ylim(ax, y_lim_pdf{ind});

    title([eqTitles{ind} ' (' eqLabels{ind} ') — Posterior Term Distributions'], ...
          'FontSize',9,'Color','k');
    xlabel('Coefficient Value','FontSize',9);
    ylabel('Probability Density','FontSize',9);

    leg = legend('Location','northeast','FontSize',9,'Box','on');
    set(leg,'TextColor','k','Color','w','EdgeColor','k');

    set(ax,'Color','w','XColor','k','YColor','k', ...
        'FontSize',9,'LineWidth',1,'TickDir','out');
    box(ax,'off'); grid(ax,'off');
    hold off;
end

%  SINDy Identified Model Trajectories

figure('Color','w','Name','ARC-SINDy Identified Model');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

axStyle = @(ax) set(ax, ...
    'Color','w','XColor','k','YColor','k', ...
    'FontSize',9,'LineWidth',1,'TickDir','out');

x_lim_id = [0, 20];
y_lim_Mx  = [-0.5,  7];
y_lim_Py  = [-0.5, 13];

% --- mRNA ---
ax1 = nexttile;
fill([0, N*dt, N*dt, 0], [y_lim_Mx(1), y_lim_Mx(1), y_lim_Mx(2), y_lim_Mx(2)], ...
    [0.85 0.85 0.85],'FaceAlpha',0.4,'EdgeColor','none','HandleVisibility','off');
hold on;
plot(t_all, x_data_all, 'k-',  'LineWidth',1.0, 'DisplayName','Sinusoidal Regression');
plot(t_all, x_id,       'b--', 'LineWidth',1.8, 'DisplayName','ARC-SINDy Model');
plot(mRNAtime, mRNA, 'ko','LineWidth',0.5, 'DisplayName','Observed Data')
title('Hes1 mRNA — SINDy Identified Trajectory','FontSize',9,'Color','k');
ylabel('Hes1 mRNA Concentration (\mug/\muL)','FontSize',9);
leg1 = legend('Location','northeast','FontSize',9,'Box','on');
set(leg1,'TextColor','k','Color','w','EdgeColor','k');
xlim(ax1, x_lim_id); ylim(ax1, y_lim_Mx);
axStyle(ax1); box(ax1,'off'); grid(ax1,'off');
hold off;

% --- Protein ---
ax2 = nexttile;
fill([0, N*dt, N*dt, 0], [y_lim_Py(1), y_lim_Py(1), y_lim_Py(2), y_lim_Py(2)], ...
    [0.85 0.85 0.85],'FaceAlpha',0.4,'EdgeColor','none','HandleVisibility','off');
hold on;
plot(t_all, y_data_all, 'k-',  'LineWidth',1.0, 'DisplayName','Sinusoidal Regression');
plot(t_all, y_id,       'r--', 'LineWidth',1.8, 'DisplayName','ARC-SINDy Model');
plot(hes1time, hes1, 'ko','LineWidth',0.5, 'DisplayName','Observed Data')
title('Hes1 Protein — SINDy Identified Trajectory','FontSize',9,'Color','k');
xlabel('Time (h)','FontSize',9);
ylabel('Hes1 Protein Concentration (\mug/\muL)','FontSize',9);
leg2 = legend('Location','northeast','FontSize',9,'Box','on');
set(leg2,'TextColor','k','Color','w','EdgeColor','k');
xlim(ax2, x_lim_id); ylim(ax2, y_lim_Py);
axStyle(ax2); box(ax2,'off'); grid(ax2,'off');
hold off;
