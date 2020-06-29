function [x, L, lambda, V, sigma2, persistence] = EigenARCH_Simulate_shrink(p, theta, T, type, n, initial_x, initial_sigma2,GARCH)
%Function to simulate the EigenARCH process

if nargin<8
    GARCH = 1;
end


x       = zeros(p, T);
lambda  = zeros(p, T);
lambda(:,1) = (20:-20/p:1); %set initial eigenvalues to some value
Lambda  = zeros(p, p,  T);   %\Lambda matrix, contains \lambda on the diagonal
sigma2  = zeros(p, p, T);
loglike = zeros(1,T);


[~, V, omega, alpha, beta, lambda_c, kappa] = EigenARCH_repar_shrink(p, n, theta, type, 'estimated', 0,GARCH);


% One period forecast
if isempty(initial_sigma2) == 1
    %rng(150); %Set fixed seed as well
    y(:,1) = V'*x(:,1);
    sigma2(:,:,1) = eye(p);
else
    %Eigenvalues
    initial_lambda = (diag((V'*initial_sigma2(:,:,end)*V)));
    initial_y = V'*initial_x(:,end);
   
    lambda(:,1)     = omega + alpha*initial_y(:).^2 + beta*initial_lambda(:);
    
    %lambda(1:n,1)     = omega + alpha*initial_y(1:n).^2 + beta*initial_lambda(1:n);
    %lambda(n+1:end,1) = lambda_c;
    Lambda(:,:,1)     = diag(lambda(:,1));
    
    %Covariance matrix
    sigma2(:,:,1) = V*Lambda(:,:,1)*V';  
    sigma2(:,:,1) = 1/2*(sigma2(:,:,1)+sigma2(:,:,1)');     
    
    %Simulate return
    [Vec, Val] = eig(sigma2(:,:,1));
    sqrtSigma = Vec*(Val.^(0.5))*Vec';
    x(:,1)   = sqrtSigma*randn(p,1);
    y(:,1) = V'*x(:,1);
end

for i=2:T
    %Eigenvalues
    lambda(:,i)   = omega+alpha*(y(:,i-1).^2)+beta*lambda(:,i-1);
    %lambda(1:n,i)   = omega+alpha*(y(1:n,i-1).^2)+beta*lambda(1:n,i-1);
    %lambda(n+1:end,i) = lambda_c;
    Lambda(:,:,i)     = diag(lambda(:,i));
    
    %Covariance matrix
    L_inv         = diag(1./lambda(:,i));
    sigma2(:,:,i) = V*diag(lambda(:,i))*V';
    sigma2(:,:,i) = 1/2*(sigma2(:,:,i)+sigma2(:,:,i)');     
    
    %Simulate return
    [Vec, Val] = eig(sigma2(:,:,i));
    sqrtSigma = Vec*(Val.^(0.5))*Vec';
    x(:,i)   = sqrtSigma*randn(p,1);
    y(:,i) = V'*x(:,i);
    
    loglike(i) = -0.5*sum(log(lambda(:,i)))-0.5*y(:,i)'*L_inv*y(:,i);
    
end

persistence = max(eig(alpha+beta)); %persistence of stochastic process

L = sum(loglike)-T*p/2*log(2*pi);


end