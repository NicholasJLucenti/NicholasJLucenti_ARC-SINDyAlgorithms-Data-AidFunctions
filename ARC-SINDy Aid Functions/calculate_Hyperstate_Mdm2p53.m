function [h_val, h_var] = calculate_Hyperstate_Mdm2p53(xi_poly, var_poly, colscale, targetScaleX)
   
    poly_scales = colscale(2:3)'; 
    hill_scale  = colscale(end);
    
    phys_xi = (xi_poly .* targetScaleX) ./ poly_scales;
    
    weights = [0.7; 0.7^2];
    force_phys = (1 * sum(phys_xi .* weights)) / 0.005;
  
    h_val = abs((force_phys * hill_scale) / targetScaleX);

    total_scaling = (weights .* (targetScaleX ./ poly_scales) .* (hill_scale / targetScaleX)).^2;
    h_var = sum(var_poly .* total_scaling) * (2/0.94)^2;
end