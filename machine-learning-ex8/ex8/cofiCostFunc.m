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

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

hypo = X * Theta';
error = (hypo - Y) .^ 2;
J_unreg = sum(sum(error .* R)) / 2;

Theta_corr = lambda * sum(sum(Theta .^ 2)) / 2;
X_corr = lambda * sum(sum(X .^ 2)) / 2;
J = J_unreg + Theta_corr + X_corr;

for i = 1:size(R,1)
	idx = find(R(i,:)== 1);
	%printf("size(idx,1) = %d\n",size(idx,1))
	%printf("size(idx,2) = %d\n",size(idx,2))
	Theta_temp = Theta(idx,:);
	%printf("size(Theta_temp,1) = %d\n",size(Theta_temp,1))
	%printf("size(Theta_temp,2) = %d\n",size(Theta_temp,2))
	Y_temp = Y (i,idx);
	%printf("size(Y_temp,2) = %d\n",size(Y_temp,2))
	X_grad(i,:) = ((X(i,:) * Theta_temp') - Y_temp) * Theta_temp;
	X_grad(i,:) = X_grad(i,:) + (lambda * X(i,:));
	endfor


for j = 1:size(R,2)
	idx = find(R(:,j)== 1);
	%printf("size(idx,1) = %d\n",size(idx,1))
	%printf("size(idx,2) = %d\n",size(idx,2))
	%Theta_temp = Theta(idx,:);
	X_temp = X(idx,:);
	%printf("size(X_temp,1) = %d\n",size(X_temp,1))
	%printf("size(X_temp,2) = %d\n",size(X_temp,2))
	Y_temp = Y (idx,j);
	%printf("size(Y_temp,1) = %d\n",size(Y_temp,1))
	%printf("size(Y_temp,2) = %d\n",size(Y_temp,2))
	Theta_grad(j,:) = ((X_temp * Theta(j,:)') - Y_temp)' * X_temp;
	Theta_grad(j,:) = Theta_grad(j,:) + (lambda * Theta(j,:));
	endfor


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
