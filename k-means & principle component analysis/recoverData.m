function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

U_reduce = U(:, 1:K);
X_rec = Z * U_reduce';		% m x K (lower dimensional dataset matrix) * k x n (eigenvector matrix transposed) produces a
							% m x n approximation of the original m x n dataset -- via
							% computing for each datapoint its projection to n-dimensional space

end
