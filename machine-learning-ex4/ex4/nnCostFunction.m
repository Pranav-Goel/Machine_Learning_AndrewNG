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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


X= [ones(m,1) X] ;
h1= sigmoid(X*Theta1') ;
h1= [ones(size(h1,1),1) h1] ;

h2= sigmoid(h1*Theta2') ;
y2=[] ;
for i=1:num_labels
    y2=[y2 y==i] ;
end
J=-(1/m)*sum(sum(y2.*log(h2)+(1-y2).*log(1-h2))) ;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
delta3= h2-y2 ;
temp1=delta3*Theta2 ;
temp1=temp1(:,2:end) ;
delta2=temp1.*sigmoidGradient(X*Theta1') ;
%delta2=delta2(:,2:end) ;
DELTA1=zeros(size(Theta1_grad)) ;
DELTA2=zeros(size(Theta2_grad)) ;
%DELTA2=[] ;
%fprintf("%f",size()
for i=1:m
    DELTA1=DELTA1+delta2(i,:)'*X(i,:) ;
end
%fprintf('%f %f %f %f %f %f',size(DELTA2,1),size(DELTA2,2),size(delta3,1),size(delta3,2),size(h1,1),size(h1,2)) ;
for i=1:m
    DELTA2=DELTA2+delta3(i,:)'*h1(i,:);
end
Theta1_grad = DELTA1/m ;
Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(lambda/m)*Theta1(:,2:end) ;
Theta2_grad=DELTA2/m ;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(lambda/m)*Theta2(:,2:end) ;
grad=[Theta1_grad(:);Theta2_grad(:)];
    
p1=sum(Theta1.^2) ;
p2=sum(Theta2.^2) ;
J=J+(lambda/(2*m))*(sum(p1(2:end))+sum(p2(2:end))) ;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
