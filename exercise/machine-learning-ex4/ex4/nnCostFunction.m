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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m,1) X];
error = 0; % 所有样本的累积误差
Delta_2 = zeros(size(Theta2));
Delta_1 = zeros(size(Theta1));

for i=1:m,
	labels = zeros(num_labels,1);
	labels(y(i)) = 1; % 10x1
	
	a_1 = X(i,:); % 1x401
	% 前向传播
	z_2 = a_1*Theta1'; % Theta1:25x401 z_2:1x25
	a_2 = sigmoid(z_2);
	a_2 = [1 a_2]; % 1x26
	z_3 = (a_2*Theta2')'; %  Theta2:10x26 z_3:10x1
	a_3 = sigmoid(z_3); % 10x1
	
	% Part 1: 计算 cost
	error += sum(-labels .* log(a_3) - (1-labels) .* log(1-a_3));
	
	% Part 2：实现反向传播
	% 误差项 delta 的维度对应节点个数，其含义是在每个神经元节点上的误差
	delta_3 =  a_3 - labels; % 10x1
	delta_2 = Theta2' * delta_3 .* a_2' .* (1-a_2'); % 26x1
	% 累积在所有样本上的梯度值
	% 梯度是每一个神经网络参数的而非神经元，维度应该与参数维度一致
	Delta_2 = Delta_2 + delta_3 * a_2;
	Delta_1 = Delta_1 + delta_2(2:end) * a_1;
end;

% Part 1: 计算 cost
% 先计算没有正则化项的 cost，若要使用正则化，只需在其后添加正则化部分对应的 cost即可
J = error/m;
% Part 1: 增加 正则化部分的cost
J += lambda/(2*m) * (sum(Theta1(:,2:end)(:).^2) + sum(Theta2(:,2:end)(:) .^ 2 ));

% Part 2：BP传播梯度
% 计算非正则化的平均梯度
D_2 = Delta_2 / m;
D_1 = Delta_1 / m;
Theta1_grad = D_1; % 25x401
Theta2_grad = D_2; % 10x26

% Part 3: 给梯度添加正则化项
Theta1_grad(:,2:end) += lambda / m .* Theta1(:,2:end);
Theta2_grad(:,2:end) += lambda / m .* Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
