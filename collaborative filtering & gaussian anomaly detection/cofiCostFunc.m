function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


predictions = X * Theta' .* R;		% vectorized computation of m x n predictions for m movies and n users
Y = Y .* R;							% element-wise product with R sets irrelevant results to zero for gradient
									% computation
errors = predictions - Y;
errors_squared = errors .* errors;	% since the cost function is squared error

L2_theta = Theta .* Theta;
L2_x = X .* X;
L2 = (sum(L2_theta(:)) + sum(L2_x(:))) * lambda;
J = (sum(errors_squared(:)) + L2) / 2;		% ta-da

X_grad = errors * Theta + lambda * X;			% vectorized gradients; errors is m x n for n users and m movies
Theta_grad = errors' * X + lambda * Theta;			% magic~~

grad = [X_grad(:); Theta_grad(:)];

end
