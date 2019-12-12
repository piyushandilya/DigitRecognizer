clear ; close all; clc

input_layer_size  = 784;  % 28x28 Input Images of Digits
#hidden_layer_size = 25;   % 25 hidden units
hidden_layer_size1 = 300;   % 300 hidden units
hidden_layer_size2 = 100;   % 100 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 (Because we have mapped "0" to label 10)

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
allData = csvread('train.csv');
allData([1],:) = [];  # Removing 1st row as it contains the headers
testData = csvread('test.csv');
testData([1],:) = [];  # Removing 1st row as it contains the headers
testData = testData./255; # Normalizing the pixel values

X = allData(1:32000, 2:size(allData, 2));
X = X./255; # Normalizing the pixel values
y = allData(1:32000, 1);

X_cv = allData(32001:42000, 2:size(allData, 2));
X_cv = X_cv./255; # Normalizing the pixel values
y_cv = allData(32001:42000, 1);

for i = 1: size(y,1)
  if(y(i) == 0)
    y(i) = 10;
  endif
endfor

for i = 1: size(y_cv,1)
  if(y_cv(i) == 0)
    y_cv(i) = 10;
  endif
endfor

m = size(X, 1);

Y = zeros(m,num_labels);
for i = 1:m
  Y(i,y(i)) = 1;
endfor

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 6: Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size1);
initial_Theta2 = randInitializeWeights(hidden_layer_size1, hidden_layer_size2);
initial_Theta3 = randInitializeWeights(hidden_layer_size2, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];


%% =============== Part 7: Implement Backpropagation ===============

fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== Part 8: Implement Regularization ===============

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);


%% =================== Part 8: Training NN ===================

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 500);
lambda = 20;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size1, ...
                                   hidden_layer_size2, ...
                                   num_labels, X, y, Y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
#[nn_params, cost] = fminunc(costFunction, initial_nn_params, options);


theta1ParameterCount = hidden_layer_size1 * (input_layer_size + 1);
theta2ParameterCount = hidden_layer_size2 * (hidden_layer_size1 + 1);
theta3ParameterCount = num_labels * (hidden_layer_size2 + 1);


Theta1 = reshape(nn_params(1: ...
                                  theta1ParameterCount), ...
                                  hidden_layer_size1, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + theta1ParameterCount): ...
                                  theta1ParameterCount + theta2ParameterCount), ...
                                  hidden_layer_size2, (hidden_layer_size1 + 1));

Theta3 = reshape(nn_params((1 + theta1ParameterCount + theta2ParameterCount): ...
                                  theta1ParameterCount + theta2ParameterCount + theta3ParameterCount), ...
                                  num_labels, (hidden_layer_size2 + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 9: Predict ===================

pred = predict(Theta1, Theta2, Theta3, X);
pred_cv = predict(Theta1, Theta2, Theta3, X_cv);
pred_test = predict(Theta1, Theta2, Theta3, testData);

for i = 1:size(pred_test,1)
  if(pred_test(i) == 10)
    pred_test(i) = 0;
  endif
endfor

submission = [[1:28000]' pred_test];
csvwrite('submission.csv', submission);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('\nCross Validation Set Accuracy: %f\n', mean(double(pred_cv == y_cv)) * 100);


##%% =================== Part 10: Check one by one ===================
##
##rp = randperm(2000);
##figure;
##for i = 1:2000
##    % Display 
##    fprintf('\nDisplaying Example Image\n');
##    vec = reshape(X_test(rp(i), :), 28, 28);
##    displayData(X_test(rp(i), :));
##
##    pred = predict(Theta1, Theta2, Theta3, X_test(rp(i),:));
##    fprintf('\nNeural Network Prediction: %d \n\n', mod(pred, 10));
##
##    % Pause with quit option
##    s = input('Paused - press enter to continue, q to exit:','s');
##    if s == 'q'
##      break
##    end
##end