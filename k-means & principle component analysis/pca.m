function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

covar_matrix = (X' * X) * m^(-1);	% compute the covariance matrix
[U, S, V] = svd(covar_matrix);		% compute eigenvectors via singular value decomposition

end
