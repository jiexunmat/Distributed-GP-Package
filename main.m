%% Loading the kin40k dataset
data = load('kin40k_data.mat');
X_train = data.x;
Y_train = data.y;
X_test = data.xtest;
Y_test = data.ytest;

%% Parameters for Distributed GP
dist_MAX_NUM_EVAL = 100;
dist_n_train = 10000;
M = 4;

%% Parameters for Full GP
full_MAX_NUM_EVAL = 100;
full_n_train = 4000;

[ dgp_rmse, dgp_time ] = runDistGP( dist_MAX_NUM_EVAL, X_train, Y_train, dist_n_train, M, X_test, Y_test );
[ fgp_rmse, fgp_time ] = runFullGP( full_MAX_NUM_EVAL, X_train, Y_train, full_n_train, X_test, Y_test );

%% Printing results
fprintf('Distributed GP got an RMSE of %f, taking %f seconds to run.\n', dgp_rmse, dgp_time)
fprintf('Full GP got an RMSE of %f, taking %f seconds to run.\n', fgp_rmse, fgp_time)