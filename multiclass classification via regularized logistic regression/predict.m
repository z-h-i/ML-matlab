function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(m, 1);

	h = @(x)(1 ./ (1 + exp(-x)));		% sigmoid
	
	X = [ones(m, 1) X];		% (5000 x 401) matrix of 5000 examples and 400 features + bias
	layer_1 = h(X * Theta1');	% (5000 x 25) matrix of 5000 outputs each for 25 hidden neurons
	layer_1 = [ones(m, 1) layer_1];		% (5000 x 26); adds empty bias vector
	layer_2 = h(layer_1 * Theta2');		% (5000 x 10) matrix of 5000 outputs each for 10 output neurons
	[raw, p] = max(layer_2, [], 2);		% p grabs the predictions for the digits


% =========================================================================


end
