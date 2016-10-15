function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C = 1;
sigma = 0.3;

C_test = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];			 % set all values for C and sigma parameters
sigma_test = C_test;

all_combinations = cell2mat(cellfun(@(x) cell2mat(x), arrayfun(@(y) arrayfun(@(z) [y z], sigma_test, 'un', false), C_test, 'un', false), 'un', false));								% generate length(C) x length(sigma) combinations of C and sigma

errors = [];

for i = 1:length(all_combinations)
	C_temp = all_combinations(i, 1);					 % grab each combination of C, sigma
	sigma_temp = all_combinations(i, 2);			
	model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));	% train model,
	prediction = svmPredict(model, Xval);				 % calculate error
	error = mean(double(prediction ~= yval));
	errors = [errors error];							 % store errors
end

[dummy best] = min(errors);								 % find the combination with smallest error

C = all_combinations(best, 1);							 % use that combination
sigma = all_combinations(best, 2);

% =========================================================================

end
