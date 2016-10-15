function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

centroids = zeros(K, n);

for i = 1:K									% for each centroid
	total = zeros(1, n);
	count = 0;
	for j = 1:m								% compute the sum then mean position of all observations assigned to it
		if i == idx(j)
			total = total + X(j, :);
			count =  count + 1;
		end
	end
	centroids(i, :) = total * (1 / count);	% reset its position to this mean position
end


end

