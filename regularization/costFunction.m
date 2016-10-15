function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


    z = X * theta;	% m X 1 vector; each element is a dotproduct of theta and x(i)
	     			% for m datapoints
    hypotheses = sigmoid(z);	% m X 1 vector of m hypotheses in sigmoid form
    cost_y1 = (y)' * log(hypotheses);   % sum of costs for y(i) = 1 
    cost_y0 = (1 - y)' * log(1 - hypotheses);   % sum of costs for y(i) = 0
	
	J = -m^(-1) * (cost_y1 + cost_y0);
	
	
	residuals = hypotheses - y;     % m X 1 vector with m prediction residuals
	
	grad = m^(-1) * X' * residuals;


% =============================================================

end
