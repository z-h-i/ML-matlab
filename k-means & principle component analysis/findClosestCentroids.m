function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

m = size(X, 1);
idx = zeros(size(X,1), 1);

distances = [];
for i = 1:m							% for each observation
	for j = 1:K						% compute the observation's distance with each centroid
		distance = (X(i, :) - centroids(j, :)) * (X(i, :) - centroids(j, :))';
		distances = [distances; distance];
	end
	[temp, idx(i)] = min(distances);		% set the observation's centroid as the closest centroid
	distances = [];
end

end

