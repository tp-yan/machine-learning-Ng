function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_tmp = [0.01,0.03,0.1,0.3,1,3,10,30];
sigma_tmp = C_tmp;
C_best = 1;
sigma_best = 1;

error_last = -1;
iterator = 20;

counter = 1;
all = length(C_tmp) * length(C_tmp);
for i=C_tmp
	for j=sigma_tmp
		fprintf('%d/%d: C:%f,sigma:%f\n',counter,all,i,j);
		model= svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j, 1e-3, iterator));
		predictions = svmPredict(model,Xval);
		error = mean(double(predictions ~= yval)); % 验证集中误分类个数作为衡量误差
		if error_last == -1 || error < error_last
			error_last = error;
			C_best = i;
			sigma_best = j;
		end
		counter += 1;
	end
end

C_best
sigma_best
			

C = C_best;
sigma = sigma_best;


% =========================================================================

end
