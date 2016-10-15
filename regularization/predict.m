function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

p = zeros(m, 1);

z = X * theta;
hypotheses = sigmoid(z);
for i=[1:length(hypotheses)]
    if hypotheses(i) < .5
	    p(i) = 0;
	end
	if hypotheses(i) >= .5
	    p(i) = 1;
	end
end

end
