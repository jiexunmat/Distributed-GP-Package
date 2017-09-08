function [varargout] = gp_distributed(hyp, inf, mean, cov, lik, x, y, xs, method)
% This function is adapted from the GPML library's gp function, and is slightly more limited.
%
% Gaussian Process inference and prediction. The gp function provides a
% flexible framework for Bayesian inference and prediction with Gaussian
% processes for scalar targets, i.e. both regression and binary
% classification. The prior is Gaussian process, defined through specification
% of its mean and covariance function. The likelihood function is also
% specified. Both the prior and the likelihood may have hyperparameters
% associated with them.
%
% Two modes are possible: training or prediction: if no test cases are
% supplied, then the negative log marginal likelihood and its partial
% derivatives w.r.t. the hyperparameters is computed; this mode is used to fit
% the hyperparameters. If test cases are given, then the test set predictive
% probabilities are returned. Usage:
%
%   training: [nlZ dnlZ, posts] = gp_distributed(hyp, inf, mean, cov, lik, x, y);
% prediction: [ymu ys2        ] = gp_distributed(hyp, inf, mean, cov, lik, x, y, xs);
%
% where:
%
%   hyp      struct of column vectors of mean/cov/lik hyperparameters
%   inf      function specifying the inference method 
%   mean     prior mean function
%   cov      prior covariance function
%   lik      likelihood function
%   x        n by D matrix of training inputs
%   y        column vector of length n of training targets
%   xs       ns by D matrix of test inputs
%
%   nlZ      returned value of the negative log marginal likelihood
%   dnlZ     struct of column vectors of partial derivatives of the negative
%            log marginal likelihood w.r.t. mean/cov/lik hyperparameters
%   ymu      column vector (of length ns) of predictive output means
%   ys2      column vector (of length ns) of predictive output variances
%   posts    An array of 'post' structs, for reuse in prediction mode, where
%            'post' is a struct representation of the (approximate) posterior.
%            It is the 3rd output in training mode.
%
% See also infMethods.m, meanFunctions.m, covFunctions.m, likFunctions.m.


% x and y are structs with M number of subsets to be distributed
% posteriors (posts) can also be passed as y during prediction mode
M = length(x);

if nargin == 7          % training mode (optimising hyperparameters)
    total_nlZ = zeros(M, 1);
    total_dnlZ = struct;
    total_dnlZ_cov = zeros(M, numel(hyp.cov));
    if ~isfield(hyp,'mean')
        hyp.mean = [];
        total_dnlZ_mean = [];
    else
        total_dnlZ_mean = zeros(M, numel(hyp.mean));
    end
    total_dnlZ_lik = zeros(M, numel(hyp.lik));
    posts = struct;

    % Parallel for loop to distribute GP experts workload to parallel workers. 
    % Change 'parfor' to 'for' if your MATLAB version doesn't have parallel processing.
    % Depending on the number of parallel workers available, 'for' may be faster than 'parfor'.
    for i=1:M
        [nlZ, dnlZ, post] = gp(hyp, inf, mean, cov, lik, x(i).data, y(i).data);
        total_nlZ(i) = nlZ;
        total_dnlZ_cov(i,:) = dnlZ.cov;
        if ~isempty(hyp.mean)
            total_dnlZ_mean(i,:) = dnlZ.mean;
        end
        total_dnlZ_lik(i,:) = dnlZ.lik;
        posts(i).data = post;
    end
    total_dnlZ.cov = sum(total_dnlZ_cov, 1)';
    total_dnlZ.mean = sum(total_dnlZ_mean, 1)';
    total_dnlZ.lik = sum(total_dnlZ_lik, 1)';
    total_nlZ = sum(total_nlZ, 1);
    varargout = {total_nlZ, total_dnlZ, posts};

elseif nargin >= 8      % prediction mode
    n_test = length(xs);
    mu = zeros(n_test,M);
    s2 = zeros(n_test,M);
    for i=1:M
        [ymu, ys2, ~, ~] = gp(hyp, inf, mean, cov, lik, x(i).data, y(i).data, xs);
        mu(:,i) = ymu;
        s2(:,i) = ys2;
    end
    if nargin == 8
        method = 'rBCM'; % default distributed GP variant used is Robust Bayesian Committee Machine
    end
    switch (method)
        case 'PoE'
            % PoE implementation
            s_star2_inv = sum((s2.^-1),2);
            s_star2_poe = 1./s_star2_inv;
            mu_poe = s_star2_poe .* sum((s2.^-1).*mu,2);
            varargout = {mu_poe, s_star2_poe};
        case 'gPoE'
            % gPoE implementation
            beta = 1/M;
            s_star2_inv = sum(beta*(s2.^-1),2);
            s_star2_gpoe = 1./s_star2_inv;
            mu_gpoe = s_star2_gpoe .* sum(beta*(s2.^-1).*mu,2);
            varargout = {mu_gpoe, s_star2_gpoe};
        case 'BCM'
            % BCM implementation
            sy2 = exp(2 * hyp.cov(end-1));      % prior variance
            tmp = (1-M)*sy2.^-1;
            s_star2_inv = sum((s2.^-1),2)+ tmp;
            s_star2_bcm = 1./s_star2_inv;
            mu_bcm = s_star2_bcm .* sum((s2.^-1).*mu,2);
            varargout = {mu_bcm, s_star2_bcm};
        case 'rBCM'
            % rBCM implementation
            sy2 = exp(2 * hyp.cov(end-1));      % prior variance
            beta = 0.5 * (log(sy2) - log(s2));
            tmp = (1 - sum(beta,2))*sy2.^-1;
            s_star2_inv = sum(beta.*(s2.^-1),2)+ tmp;
            s_star2_rbcm = 1./s_star2_inv;
            mu_rbcm = s_star2_rbcm .* sum(beta.*(s2.^-1).*mu,2);
            varargout = {mu_rbcm, s_star2_rbcm};
    end
end
end