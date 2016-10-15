function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X, 2) + 1;
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


X = [ones(m, 1), X];		% m x (n + 1); m datapoints, n features + 1 bias
layer_1_outputs = sigmoid(X * Theta1');		% m x n; m datapoints, n neurons
layer_1_outputs = [ones(m, 1), layer_1_outputs]; 	% m x (n + 1); m points, n neurons + 1 biase
layer_2_outputs = sigmoid(layer_1_outputs * Theta2');		% m x n; n total outputs per datapoint
y_binary = [];

for i = 1:num_labels
	y_binary = [y_binary y == i];		% m x L; m datapoints, L classes
end

j = 0;
for i = 1:m
	j = j + log(layer_2_outputs(i, :)) * y_binary(i, :)' + log(1 - layer_2_outputs(i, :)) * (1 - y_binary(i, :)');
end
J = -m^(-1) * j;

Theta1_sqr = Theta1(:, 2:end) .* Theta1(:, 2:end);
Theta2_sqr = Theta2(:, 2:end) .* Theta2(:, 2:end);
r = sum(sum(Theta1_sqr)) + sum(sum(Theta2_sqr));

J = J + lambda * (2*m)^(-1) * r;

Theta2_delta = layer_2_outputs - y_binary;
Theta1_delta = Theta2_delta * Theta2(:, 2:end) .* layer_1_outputs(:, 2:end) .* (1 - layer_1_outputs(:, 2:end));

Theta2_grad = (Theta2_delta' * layer_1_outputs) * m^(-1);
Theta1_grad = (Theta1_delta' * X) * m^(-1);

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + m^(-1) * lambda * Theta2(:, 2:end);
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + m^(-1) * lambda * Theta1(:, 2:end);



grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
