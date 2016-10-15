function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2) + 1;

all_theta = zeros(num_labels, n);

% Add ones to the X data matrix
X = [ones(m, 1) X];

	theta_0 = zeros(n, 1);
	options = optimset('GradObj', 'on', 'MaxIter', 50);
	
	for i = 1:num_labels
		all_theta(i, :) = fmincg( @(theta)(lrCostFunction(theta, X, (y == i), lambda)), theta_0, options);
	end	

% =========================================================================


end
