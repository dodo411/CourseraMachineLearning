function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X*theta;
hth = sigmoid(z);
log_hth1 = log(hth);
log_hth2 = log(1-hth);
theta2 = theta(2:end);
J = (-y'*log_hth1 - (1-y')*log_hth2)/m +lambda/2/m*(theta2'*theta2);

grad_temp = ((hth - y)'*X)/m;
grad(1) = grad_temp(1);
grad(2:end) = grad_temp(2:end)' + theta2*lambda/m;







% =============================================================

end
