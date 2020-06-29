%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%      INFORMATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This matlab file contains the code necessary to compute the standard
% errors for the Spectral Targeting estimator of the EigenGARCH model as
% presented in Hetland (2020).  
% 
% This file contain the code to estimate the model and include four specific functions:
% 'asymp_var'           - computes the asymptotic variance and standard errors of the ith equation  
% 'finite_k_mom'        - is used to check the moment requirement for finite
% fourth order moments. Implemented assuming Gaussian innovations
% 'loglikelihood'       - the i'th log-likelihood function
% 'loglikelihood_cont'  - the vector of log-likelihood contributions.
%
% Furthermore, the file uses some functions saved in the "Library" folder,
% which are used to construct graphs. We use the DERIVESTsuite to compute
% the (numerical) first and second order derivatives
%
% Author: Simon Hetland
% e-mail: bhp240@ku.dk
% 16 June 2020


%% Housekeeping
clc
clear
format longG

addpath('Library/')
addpath('Library/DERIVESTsuite')
addpath('../../')

%SP500 Sectors
Data     = readtable('SP500_sectors.xlsx');   
T_start  = size(Data,1)-8*252;
T_end    = size(Data,1);
   
%% Load data and specify options
type      = 'full';                                        %'diagonal' or 'full'
var_est   = 'targeting';                                   %'targeting' or 'estimated'

GARCH = 1;                                            % ARCH(1) or GARCH(1,1) model
p    = 2;                                             % # of variables
 
%Construct matrix of data
x     = table2array(Data(T_start:T_end,2:1+p))';            
dates = table2array(Data(T_start:T_end,1))';
T     = length(x);      

npar = 0; %Number of GARCH parameters (not including first step estimation of unconditional covariance matrix)
if isequal(type,'full')          npar = npar + p; 
else                             npar = npar + 1; end
if GARCH ==1                     npar = npar + 1; end
if isequal(var_est, 'estimated') npar = npar + 1; end
%% Summary statistics
fprintf('\nSummary statistics (average return and vol. in percent p.a.) \n')
fprintf('Correlations') 
corr_data = corr(x')

fprintf('Minimum and  maximum correlations')
[min(vech_subdiag(corr_data)), max(vech_subdiag(corr_data))]

fprintf('Average return and volatility (percent p.a.)') 
[mean(x')*252; std(x')*sqrt(252)]

MGARCH_graphics(x,[],dates, 'desc'); % 'all', 'desc', 'res', 'vol'
 
%% 1. step estimator - unconditional eigenvalues and -vectors
%Estimate unconditional covariance matrix
H_uncon = cov(x');

%Spectral decomposition
[V,l] = eigs(H_uncon,p);
l=diag(l);

%Identifying normalization
for i=1:p
   if V(i,i)<0
       V(:,i)=V(:,i)*(-1);
   end
end

%First step estimator
theta_step1 = [l;V(:)];

%Construct rotated returns
y = V'*x(:,:);

%Summary statistics wrt. 1. step estimator
lambda_per = 100.*l./sum(l);

fprintf('\nEigenvalues of unconditional covariance matrix are (percent)\n')
fprintf('%3.2f ',lambda_per)
 
%% 2. step estimator - univariate (augmented) GARCH(1,1)
%Matrices to contain estimated variances and residuals
lambda = zeros(p,T);
z_hat = zeros(p,T);
L = zeros(p,3);
theta_i = zeros(p,npar);

%Matrices to contain estimated parameters
if isequal(type, 'diagonal')        a = zeros(p,1); else a = zeros(p,p); end
if GARCH==1                         b = zeros(p,1); else b=0;            end
w = zeros(p,1); 

for j=1:p
    %Initial values for estimation
        if isequal(type, 'diagonal') 
            theta0 = [0.05, 0.75, 0.8];                   %[a, b, w]
        elseif isequal(type, 'full') 
            theta0 = [ones(1,p)*0.02, 0.75, 0.8];         %[a1, ..., ap, b, w]
            theta0(j) = 0.05; 
        end
        if isequal(var_est,'targeting') theta0 = theta0(1:end-1);  end %remove w
        if GARCH==0 theta0=theta0(1:end-1);                        end %renove b

    %Evaluate log-likelihood function to check if everything works
    [ll, lambda(j,:), z_hat(j,:)] = loglikelihood(x,theta0,type,var_est,GARCH,j,0);

    %Settings for numerical optimization
    %options = optimset('Display','iter','PlotFcns',@optimplotfval,'UseParallel', true, 'TolFun',1e-8, 'TolX',1e-10, 'MaxFunEvals',300000, 'MaxIter', 1000000);
    options = optimset('Display','iter','PlotFcns',@optimplotfval,'UseParallel', true,'Algorithm', 'sqp', 'TolFun',1e-8, 'TolX',1e-10, 'MaxFunEvals',300000, 'MaxIter', 1000000);

    %Bounds for numerical optimization
    lb = [zeros(size(theta0))];
    ub = [ones(size(theta0))*3];

    %%Numerical optimization of log-likelihood function
    [theta_j, ll_j, exit_j, output_j, ~, grad_j, hess_j] = ...
        fmincon(@(coef) -loglikelihood(x,coef(), type, var_est, GARCH, j,0),...
        theta0, [],[],[],[], lb, ub, [], options);

    %Vector containing log-likelihood value, foc and soc (should be 0 and 1 respectively)
    L(j,:) = [-T*ll_j, output_j.firstorderopt, min(eig(hess_j))>0]; 

    %Retrieve estimated variance and residual
    [~, lambda(j,:), z_hat(j,:)] = loglikelihood(x,theta0,type,var_est,GARCH,j,0);
    
    %Save parameters in matrices
    if isequal(type, 'diagonal')        a(j)   = theta_j(1);     tal = 2; 
    else                                a(j,:) = theta_j(1:p);   tal = p+1;    end
    if GARCH == 1                       b(j)   = theta_j(tal);   tal = tal+1;
    else                                b(j)   = 0;                         end
    if isequal(var_est,'estimated')     w(j)   = theta_j(tal);   tal = tal+1;
    else
         if isequal(type, 'diagonal')   w(j)   = (1-b(j))*l(j)-a(j,:)*l(j);
         else                           w(j)   = (1-b(j))*l(j)-a(j,:)*l;    end 
    end
    
    %Save estimated parameters
    theta_i(j,:) = theta_j; 
end
%Second step estimator
theta_step2 = [a(:);b(:)];

%Joint parameter vector
theta = [theta_step1; theta_step2];

fprintf('Univariate log-likelihoods, FOC and SOC:')
L
fprintf('\nEstimated parameters:')
w
a
b

%% Compute conditional covariance matrices and joint log-likelihood
% Joint likelihood
L_joint = sum(L(:,1))

% Estimated covariances
H = zeros(p,p,T);
for i=1:T
    H(:,:,i) = V*diag(lambda(:,i))*V';
end

%Graphics of estimated eigenvalues and covariance + residual diagnostics
EigenARCH_graphics(y,lambda,dates,1);
MGARCH_graphics(x,H,dates, 'all'); % 'all', 'desc', 'res', 'vol'

%% Check existence of fourth order moment
k = 4; %Moment of X_t 

%Simulate (Gaussian innovations) for k>2, closed form solution for k=2. 
[mom_k, mom_k_ci] = finite_k_mom(x,[a(:);b(:)], k, GARCH, type, var_est) %mom_k and mom_k_ci should be <1 for k=4


%% Inference
se_i = zeros(p,npar);

for j=1:p %Loop for the p univariate equations
    theta_j = [theta_step1(:); theta_i(j,:)'];
    se_j = asymp_var(x, theta_j, type, GARCH, j);

    se_i(j,:) = se_j(p^2+p+1:end); %S.e.'s for (G)ARCH parameters for the j'th equation
end
se_cov = se_j(1:p^2+p);             %S.e.'s for the eigenvalues and vectors
        

 %% Print output nicely
 disp('Estimates, se, t-stat for step 1 ') 
 disp([[l; V(:)], se_cov, [l; V(:)]./se_cov])
 
 for i=1:p    
    disp('Estimates, s.e., t-stat for equation ') 
    disp(i)
    disp([theta_i(i,:)',se_i(i,:)', theta_i(i,:)'./se_i(i,:)' ])
 end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [se_sample] = asymp_var(x, theta,type, GARCH, j)
%Inputs
% x             - pxT matrix of data
% theta         - the parameters of the i'th equation (called theta^{(i)} in the paper)
% type          - "full" or "diagonal", diagonal if the A matrix restricted to be diagonal
% GARCH         - 1 to include GARCH term, 0 for ARCH
% j             - the j'th equation in the EigenGARCH

%Outputs:
% se_sample     - the (sample) standard errors for theta^{(i)}

%Ensure data is stored in rows
if size(x,2)<size(x,1)
    x=x';
end

%Constants
T = length(x); %Number of observations in x.
p = size(x,1); %Number of astheta_set returns in x.

%Eigenvalues and eigenvectors
l = theta(1:p);
V = reshape(theta(p+1:p+p^2),p,p);
H = V*diag(l)*V';

D = zeros(p,p^2);
for i =1:p
   D(i,i+(i-1)*p)=1; 
end

%Jacobian of loglikelihood contributions
Jac = jacobianest(@(coef) -loglikelihood_cont(x,coef(), type, 'targeting', GARCH, j, 1),theta);
score = Jac(:,p^2+p+1:end)'; %Pick out the right elements (relating to ARCH+GARCH parameters)

npar=p^2+p; %# parameters in l and V
if GARCH==1 npar = npar+1; end % + GARCH parameters of equation i
if isequal(type,'full') npar=npar+p; else npar=npar+1; end
  
Omega = zeros(npar, npar);
omega = zeros(npar,1);
 
 %Loop to compute Omega^(i)
 for t=2:T    
    tmp = x(:,t)*x(:,t)';
    tmp = tmp(:)-H(:);     
          
    %Eq. (4.3)
    %Fill in eigenvalues
    omega(1:p) = D*kron(V,V)*tmp;  
    %Fill in eigenvectors
    for i=1:p
       omega(p+1+(i-1)*p:p+(i-1)*p+p) = kron(V(:,i)',pinv(l(i)*eye(p)-H))*tmp;
    end
    %Fill in score of log-likelihood function
    omega(p+(p-1)*p+p+1:end) = score(:,t);    
     
    %Compute Omega
    Omega = Omega + omega*omega';
        
end
Omega = 1/T*Omega; %%Eq. (4.3)

%Compute numerical hessian, 1/T sum d^2 l/dpdp'
Hess = hessian(@(coef) -loglikelihood(x,coef(), type, 'targeting', GARCH,j, 1),theta); %Eq. (4.2)

%Construct J and K matrices
J_inv = inv(Hess(p^2+p+1:end,p^2+p+1:end)); % Eq. (4.2): (1/T sum d^2 L/dkdk')^(-1)
K = Hess(p^2+p+1:end,1:p^2+p);              % Eq. (4.2): 1/T sum d^2 L/dkdg'

%Construct sigma
Sigma = [eye(p^2+p), zeros(p^2+p,npar-p^2-p); % Eq. (4.1)
            -J_inv*K, -J_inv];

%Compute asymptotic variance-covariance matrix        
cov_sample = Sigma*Omega*Sigma'/T; %Divide by T to get \hat\theta ~ N(\theta_0, cov/T) 

%Compute standard errors of 1 and 2 step estimates.
se_sample = sqrt(diag(cov_sample));

end

function [eig_H_k, eig_H_se] = finite_k_mom(x, param, k, GARCH, type, var_est)
%Inputs
% x             - pxT matrix of data
% param         - the vector of all parameters, ordered as param = [A(:), diag(B)] 
% k             - the moment which existence you want to check, e.g. k=2 for E||X_t||^2<\infty
% GARCH         - 1 to include GARCH term, 0 for ARCH
% type          - "full" or "diagonal", diagonal if the A matrix restricted to be diagonal
% var_est       - targeting or estimated, always set to targeting in this application
%Outputs
% eig_H_k       - a scalar, if eig_H_k<1, the k'th moment is finite
% eig_H_se      - 5% and 95% confidence bands, both should be less than 1

if k<2
    error('k must be 2 or higher')
end

k=k/2; %x^k -> l^k/2

%Ensure data is stored in rows
if size(x,2)<size(x,1)
    x=x';
end

[p,T] = size(x);

[V,l] = eigs(cov(x'));

%Identifying normalization
for i=1:p
   if V(i,i)<0
       V(:,i)=V(:,i)*(-1);
   end
end

y = V'*x;

tal=1;
if isequal(type,'diagonal')
    a = diag(param(tal:tal+p-1));
    tal=tal+p;
elseif isequal(type,'full')
    a = reshape(param(tal:tal+p^2-1),p,p);
    tal=tal+p^2;
end

if GARCH==1
    b = diag(param(tal:tal+p-1));
    tal = tal+p-1;
else
    b = zeros(p,p);
end

if isequal(var_est,'estimated')
   w = param(tal:tal+p-1);
   tal = tal+p;
else
   w = (eye(p)-a-b)*diag(l); 
end

if k==1
    eig_H_k=max(eigs(a+b)); 
    eig_H_se = 0;
else
    N = 399;
    lambda  = zeros(p,T);
    H       = zeros(p,p,T);
    H_k = 0;
    H_k_vec = zeros(p^k^2,N);
    for n=1:N        
        for t=2:T
            z = randn(p,1);
            lambda(:,t) = w+a*y(:,t-1).^2+b*lambda(:,t-1);
            H(:,:,t) = a*diag(z.*z)+b;                %SRE coefficient
            tmp =kron(H(:,:,t), H(:,:,t));
            for i=3:k
                tmp = kron(tmp, H(:,:,t));            
            end
            H_k = H_k + tmp;
        end
        H_k = 1/T*H_k;
        H_k_vec(:,n) = H_k(:);
    end
    H_k_5    = reshape(quantile(H_k_vec',[0.05])',p^k,p^k);
    H_k_95   = reshape(quantile(H_k_vec',[0.95])',p^k,p^k);
    H_k_mean = reshape(mean(H_k_vec')',p^k,p^k);
    
    eig_H_k  = max(eigs(H_k_mean));
    eig_H_se = [max(eigs(H_k_5)),max(eigs(H_k_95))];
end

end

function [L, h, z_hat] = loglikelihood(x, param, type, var_est, GARCH, j, se_est)
%Inputs
% x             - pxT matrix of data
% param         - the vector of parameters for the j'th equation
% type          - "full" or "diagonal", diagonal if the A matrix restricted to be diagonal
% var_est       - targeting or estimated, always set to targeting in this application
% GARCH         - 1 to include GARCH term, 0 for ARCH
% j             - which series of rotated returns we are modeling
% se_est        - 0 or 1, set to 0 in normal estimation, also used by  "asymp_var" function
%Outputs
% L             - the log-likelihood value for retunr j
% h             - estimated conditional eigenvalues
% z_hat         - standardized residuals (for diagonistics etc.)

[loglik, h, z_hat] = loglikelihood_cont(x, param, type, var_est, GARCH, j, se_est);
T = length(x);
L = 1/T*sum(loglik);
end

function [loglik, h, z_hat] = loglikelihood_cont(x, param, type, var_est, GARCH, j, se_est)
%Inputs
% x             - pxT matrix of data
% param         - the vector of parameters for the j'th equation
% type          - "full" or "diagonal", diagonal if the A matrix restricted to be diagonal
% var_est       - targeting or estimated, always set to targeting in this application
% GARCH         - 1 to include GARCH term, 0 for ARCH
% j             - which series of rotated returns we are modeling
% se_est        - 0 or 1, set to 0 in normal estimation, also used by  "asymp_var" function
%Outputs
% L             - the log-likelihood value for retunr j
% h             - estimated conditional eigenvalues
% z_hat         - standardized residuals (for diagonistics etc.)

%Check inputs
switch nargin
    case 6
        se_est = 0;
    case 7    
        %do nothing
    otherwise
        error('Error: Not enough inputs')
end

%Ensure data is stored in rows
if size(x,2)<size(x,1)
    x=x';
end

%Constants
T = length(x); %Number of observations in x.
p = size(x,1); %Number of asset returns in x.
tal = 0;       %Counting variable

if se_est==1   
    %Reconstructing eigenvalues and vectors (if computing s.e.'s)
    l = diag(param(1:p));                     
    tal = tal+p+1;
    V = reshape(param(tal:tal+p^2-1),p,p);    
    tal = tal+p^2;
else
    %Estimate unconditional covariance matrix (for estimation)
    H = cov(x');
    tal = 1;
   
    %Recover eigenvalues and eigenvectors
    [V,l] = eigs(H,p);
    
    %Identifying normalization
    for i=1:p
        if V(i,i)<0
            V(:,i)=V(:,i)*(-1);
        end
    end       
end

%Construct rotated returns
y = V'*x;

%NO REPARAMETERIZATION - SHOULD BE ESTIMATED USING CONSTRAINED OPTIMIZATION
%PARAMETER VECTOR SHOULD BE,
% param = [alpha, beta, omega] if estimated intercept
% param = [alpha, beta] if covariance targeting
% alpha should be either px1 or 1x1 if full or diagonal specification
% respectively.

if isequal(type, 'diagonal')
    a = zeros(1,p);
    a(j) = param(tal);
    tal = tal+1;
elseif isequal(type,'full')
    a = reshape(param(tal:tal+p-1),1,p);
    tal = tal+p;
end

if GARCH==1
   b = param(tal);
   tal = tal+1;
elseif GARCH==0 
   b=0;
end

if isequal(var_est, 'targeting')
    w = (1-b)*l(j,j)-a*diag(l);
else %isequal(var_est,'estimated')
    w = param(tal);
    tal = tal + 1; 
end 
    
%Vectors to contain stuff
loglik    = zeros(1, T);
h         = zeros(1, T);
h(1)      = l(j,j); %initiate in unconditional variance
z_hat     = zeros(1, T);

%Loop to compute recursion of variances, residuals, and likelihood
%contributions.
for i = 2:T     
    %Volatility
    h(i)  = w+a*y(:,i-1).^2+b*h(i-1);
    
    %Residual
    z_hat(i)   = y(j,i)/sqrt(h(i));
    
    %Log likelihood contribution
    loglik(i) = - 1/2*log(2*pi) -1/2 * (log(h(i)) + y(j,i)^2/h(i));   
end

end
