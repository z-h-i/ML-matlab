function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	predictions = X * theta;				% m x 1 vector for m predictions, 1/input
    errors = predictions - y;				% m x 1 vector for m errors, 1/prediction
    error_feature_sums = X' * errors;		% n x 1 vector for n features
											% represents sum(error * feature) for 
											% each data point/input
											
	scalar = alpha / m;						% learning rate * (1 / m)
	update = scalar * error_feature_sums;	% n x 1 vector of n partial derivatives in
											% (learning rate * gradient)
	theta_new = theta - update;				% n x 1 vector for n new features
	theta = theta_new;						% simultaneous update





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
