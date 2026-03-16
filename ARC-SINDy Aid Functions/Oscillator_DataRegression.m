%% preSINDy_Sync.m
% Use this script to clean and synchronize raw p53/Mdm2 data before SINDy
clear; clc; close all;

%% 1. Load your raw "disarray" data
% Variables expected: p53_raw, p53_time, Mdm2_raw, Mdm2_time
load('p53_set1.mat');
load('Mdm2_set1.mat');
p53_time = p53_set1(:,1);
Mdm2_time = Mdm2_set1(:,1);
p53_raw = p53_set1(:,2);
Mdm2_raw = Mdm2_set1(:,2);

%% 2. Define Synchronized Target Grid
dt_target = 0.05; 
t_start = max(min(p53_time), min(Mdm2_time));
t_end   = min(max(p53_time), max(Mdm2_time));
t_common = (t_start:dt_target:t_end)';

%% 3. Regression Settings
% 'sin5' fits: y = sum_{i=1}^5 a_i * sin(b_i*t + c_i)
ft = fittype('sin4'); 
opts = fitoptions('Method', 'NonlinearLeastSquares');
opts.Display = 'Off';


%% 4. Execute 5-Term Sine Regression with Mean Centering
% p53 (x) centering
mu_x = mean(p53_raw);
p53_centered = p53_raw - mu_x;

fprintf('Performing 5-term Sinusoidal Regression on p53 (x)...\n');
[fit_x, gof_x] = fit(p53_time(:), p53_centered(:), ft, opts);
% Evaluate and add mean back
p53_cleanData1 = feval(fit_x, t_common) + mu_x;

% Mdm2 (y) centering
mu_y = mean(Mdm2_raw);
Mdm2_centered = Mdm2_raw - mu_y;

fprintf('Performing 5-term Sinusoidal Regression on Mdm2 (y)...\n');
[fit_y, gof_y] = fit(Mdm2_time(:), Mdm2_centered(:), ft, opts);
% Evaluate and add mean back
Mdm2_cleanData1 = feval(fit_y, t_common) + mu_y;

%% 6. Diagnostic Visualization
figure('Color', 'k', 'Name', 'Sinusoidal Synchronization Results');
plot(p53_time, p53_raw, 'w.', 'MarkerSize', 8); hold on;
plot(t_common, p53_cleanData1, 'r-', 'LineWidth', 2);
title(['p53 Alignment (R^2: ', num2str(gof_x.rsquare, 3), ')'], 'Color', 'w');
legend('Raw Disarray', 'Sinusoidal Fit'); grid on;
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');
hold off;

figure('Color', 'k', 'Name', 'Sinusoidal Synchronization Results');
plot(Mdm2_time, Mdm2_raw, 'w.', 'MarkerSize', 8); hold on;
plot(t_common, Mdm2_cleanData1, 'c-', 'LineWidth', 2);
title(['Mdm2 Alignment (R^2: ', num2str(gof_y.rsquare, 3), ')'], 'Color', 'w');
legend('Raw Disarray', 'Sinusoidal Fit'); grid on;
xlabel('Time', 'Color', 'w');
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');

% save('p53_cleanData1','p53_cleanData1'); save('Mdm2_cleanData1','Mdm2_cleanData1')