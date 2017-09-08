%% HELPER FUNCTION
% This function splits the data up to M subsets, for M local GP experts to train on.
% In case the dataset cannot be divided into M even subsets, the last subset will be smaller.
% Also, the subsets are randomly chosen. This is the method used by Deisenroth and Ng.
function [X_split, Y_split] = split_data(X, Y, M)
    n = length(X);
    assert (n == length(Y));
    X_split = struct;
    Y_split = struct;
    set_size = ceil(n/M);

    % Randomising the data so as to choose random subsets.
    random_order = randperm(n);
	X_rand = X(random_order,:);
	Y_rand = Y(random_order,:);

    for i=1:M
        start_index = (i-1)*set_size + 1;
        end_index = min(i*set_size, n);
        X_split(i).data = X_rand(start_index:end_index, :);
        Y_split(i).data = Y_rand(start_index:end_index, :);
    end
end