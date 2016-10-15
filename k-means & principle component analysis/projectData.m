function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

U_reduce = U(:, 1:K);			% select K dimensions to reduce data to
Z = X * U_reduce;				% calculate projection of all datapoints onto K dimensions via
								% m x n (dataset matrix) * n x K (eigenvector matrix)

end
