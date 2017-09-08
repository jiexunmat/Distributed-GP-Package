function [ rmse, time ] = runFullGP( MAX_NUM_EVAL, X_train, Y_train, n_train, X_test, Y_test )
%% Setting up data
% Taking n_train random data points from the entire dataset
random_order = randperm(n_train);
X_train = X_train(random_order,:);
Y_train = Y_train(random_order,:);

%% Setting up cov, mean, inf functions
% Initialise guess: logtheta0 (from Swordfish)
stdX = std(X_train)';
stdX( stdX./abs(mean(X_train))' < 100*eps ) = Inf; % Infinite lengthscale if range is negligible
logtheta0 = log([stdX; std(Y_train); 0.05*std(Y_train)]);

% Use a simple covariance function that is the sum of a squared exponential covariance and a noise
hyp=[];inf=[];lik=[];cov=[];
cov = {@covSum, {@covSEard,@covNoise}}; 
hyp.cov = logtheta0;
meanfunc = [];
lik = {@likGauss};
hyp.lik = logtheta0(end);
inf = @infGaussLik;

%% Optimise hyperparameters for full GP
fprintf('Optimising hyperparameters for full GP...\n')
tic;
hyp = minimize_minfunc(hyp,@gp,-MAX_NUM_EVAL,inf,meanfunc,cov,lik,X_train,Y_train);
time = toc;

% Save posterior from after training the hyperparameters
[~, ~, post] = gp(hyp, inf, meanfunc, cov, lik, X_train, Y_train);

%% Generate predictions on test set
if nargin == 6
    [ymu,ys2] = gp(hyp,inf,meanfunc,cov,lik,X_train,post,X_test);
    rmse = computeRMSE(ymu,Y_test);
else
    rmse = 0;
end
end

