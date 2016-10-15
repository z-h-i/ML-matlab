function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

J = 0;
grad = zeros(size(theta));


    z = X * theta;	% m X 1 vector; each element is a dotproduct of theta and x(i)
	     			% for m datapoints
    hypotheses = sigmoid(z);	% m X 1 vector of m hypotheses in sigmoid form
    cost_y1 = (y)' * log(hypotheses);   % sum of costs for y(i) = 1 
    cost_y0 = (1 - y)' * log(1 - hypotheses);   % sum of costs for y(i) = 0
	penalty = lambda * (2*m)^(-1) * theta(2:n)' * theta(2:n);  % regularization penalty
	J = -m^(-1) * (cost_y1 + cost_y0) + penalty;
	
	
	residuals = hypotheses - y;     % m X 1 vector with m prediction residuals
	penalty_grad = lambda * m^(-1) * theta(2:n);    % penalty in the gradient
	
	grad(1) = m^(-1) * sum(residuals);    % intercept doesn't have a penalty
	grad(2:n) = m^(-1) * X(:, 2:n)' * residuals + penalty_grad;

end
