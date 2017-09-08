function [ rmse, time ] = runDistGP( MAX_NUM_EVAL, X_train, Y_train, n_train, M, X_test, Y_test )
%% Setting up data
random_order = randperm(n_train);
X_train = X_train(random_order,:);
Y_train = Y_train(random_order,:);
[X_split, Y_split] = split_data(X_train, Y_train, M);

%% Initialize hyp, cov, mean, lik
% Initialise guess: logtheta0 (from Swordfish)
stdX = std(X_train)';
stdX( stdX./abs(mean(X_train))' < 100*eps ) = Inf; % Infinite lengthscale if range is negligible
logtheta0 = log([stdX; std(Y_train); 0.05*std(Y_train)]);

% Using a simple covariance function that is the sum of a squared exponential covariance kernel and a noise
hyp=[];inf=[];lik=[];cov=[];
cov = {@covSum, {@covSEard,@covNoise}}; 
hyp.cov = logtheta0;
meanfunc = [];
lik = {@likGauss};
hyp.lik = logtheta0(end);
inf = @infGaussLik;

%% Optimise hyperparameters for distributed GP
fprintf('Optimising hyperparameters for distributed GP...\n')
tic;
hyp = minimize_minfunc(hyp,@gp_distributed,-MAX_NUM_EVAL,inf,meanfunc,cov,lik,X_split,Y_split);
time = toc;

% Save posteriors from after training the hyperparameters
[~, ~, posts] = gp_distributed(hyp, inf, meanfunc, cov, lik, X_split, Y_split);

%% Generate predictions on test set
if nargin == 7
    [ymu_dgp,ys2_dgp] = gp_distributed(hyp,inf,meanfunc,cov,lik,X_split,posts,X_test,'rBCM');
    rmse = computeRMSE(ymu_dgp,Y_test);
else
    rmse = 0;
end
end