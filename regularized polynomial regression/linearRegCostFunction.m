function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

predictions = X * theta;		% m x n; for m examples and n features
cost = predictions - y;
RSS = cost' * cost;				% residual sum of squares
regularization_factor = lambda * theta(2:end)' * theta(2:end);		% sum of theta squared
J = (RSS + regularization_factor) / (2 * m);

RSS_grad = X' * cost;
reg_grad = theta(2:end) * lambda;				% lambda * theta penalty for all theta except 'intercept' theta
grad_0 = RSS_grad(1);
grad_rest = RSS_grad(2:end) + reg_grad;
grad = m^(-1) * [grad_0 ; grad_rest];


% =========================================================================

grad = grad(:);

end
