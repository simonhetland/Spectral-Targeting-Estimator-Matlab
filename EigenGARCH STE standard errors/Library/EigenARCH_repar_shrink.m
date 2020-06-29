function [gamma, V, omega, alpha, beta, lambda_c, kappa, param_rotation] = EigenARCH_repar_shrink(p, n, param, type, var_est, GJR, GARCH)
% Function to reparameterize the parameter-vector to the matrices

if nargin<5 || isequal(var_est, [])
    var_est = 'estimated';
end
if nargin<6 || isequal(GJR, [])
    GJR = 0;
end
if nargin<7 || isequal(GARCH, [])
    GARCH = 1;
end



tal=1;

param_rotation = zeros(p*(p-1)/2,1);
if n>p%n<p %Repeat rotation parameters p/2 times 
     j=1;
     
   for i=1:ceil(p/2)
      param_rotation(j:j+p-2) = param(tal:tal+p-2);
      j=j+p-1; 
   end
   tal=tal+p-1;
    
else %Do not repeat rotation parameters
    param_rotation = param(tal:tal + p*(p-1)/2-1);
    %param_rotation = exp(param_rotation)./(1+exp(param_rotation)).*pi/2;
    tal=tal+p*(p-1)/2;
end

V = rotation(param_rotation,p); %Rotation matrix
gamma = [V(:)];

if isequal(var_est, 'estimated')
    omega = (param(tal:tal+n-1));
    tal=tal+n;
    gamma = [gamma; omega];
    
end

if isequal(type,'scalar')
    alpha = diag(param(tal).^2*ones(n,1));
    tal = tal + 1;
    
    if GARCH == 1
        beta = diag(param(tal).^2*ones(n,1));
        tal = tal + 1;
        gamma = [gamma; alpha(1); beta(1)];
    else 
        beta = zeros(n,n);
    end
    if GJR ==1
        kappa = diag(param(tal).^2*ones(n,1));
        tal = tal +1;
        gamma = [gamma; kappa(1)];
    end
    
elseif isequal(type,'diagonal')
    alpha =zeros(p,p);
    alpha(1:n,1:n)= diag(param(tal:tal+n-1));
    %alpha = diag(param(tal:tal+n-1).^2);
    tal = tal + n;
    
    beta = zeros(p,p);
    if GARCH ==1
        beta(1:n,1:n) = diag(param(tal:tal+n-1));
        %beta = diag(param(tal:tal+n-1).^2);
        tal = tal + n;
        gamma = [gamma; diag(alpha); diag(beta)];
    else 
        beta = zeros(n,n);
    end
    
    if GJR ==1
        kappa = diag(param(tal:tal+n-1).^2);
        tal = tal + n;
        gamma = [gamma; diag(kappa)];
    end

elseif isequal(type,'full')
    alpha = zeros(p,p);
    alpha(1:n,1:n)     = vec2mat(param(tal:tal+n^2-1),n)';
    tal = tal + n*n;    

    % GARCH-X type thing
    alpha(1:n,n+1:end) = vec2mat(param(tal:tal+n*(p-n)-1),n,p-n)'; 
    tal = tal + n*(p-n);    

    
    beta = zeros(p,p);
    if GARCH ==1      
        %beta(1:n,1:n) = diag(param(tal:tal+n-1));
        %beta = diag(param(tal:tal+n-1).^2);
        %tal = tal + n;
        %gamma = [gamma; alpha(:); diag(beta)];     
        
        beta(1:n,1:n) = vec2mat(param(tal:tal+n^2-1),n)';
        tal = tal + n*n;
        gamma = [gamma; alpha(:); beta(:)];        
    end
    
    if GJR ==1
        kappa = zeros(p,p);
        kappa(1:n,1:n) = vec2mat(param(tal:tal+n^2-1).^2,n)';
        tal = tal + n*n;
        gamma = [gamma; kappa(:)];
    end    
end    

if isequal(var_est, 'targeting')
    omega = [];
end

if n<p && isequal(var_est, 'estimated')
   lambda_c = exp(param(tal:end));
   omega = [omega; lambda_c];
   
   gamma = [gamma; lambda_c];
else
    lambda_c = [];
end

if GJR ==0
    kappa = [];
end

end