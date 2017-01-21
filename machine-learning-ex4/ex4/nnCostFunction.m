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

% Calculate a1
a1 = [ones(m, 1) X];

% Second layer a2
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];

% Third layer (h)
z3 = a2 * Theta2';
h = sigmoid(z3);

% Calculate the cost of each output
for k = 1:num_labels
  yk = y == k; % We only need y in the k activation
  hk = h(:, k); % The hipothesis in the k activation
  Jk = (1/m) * sum(-yk .* log(hk) - (1 - yk) .* log(1 - hk));
  J = J + Jk;
end


% Compute the regularization term
regTerm1 = sum(sum(Theta1(:, 2:end) .^ 2));
regTerm2 = sum(sum(Theta2(:, 2:end) .^ 2));
regTerm = lambda / (2 * m) * (regTerm1 + regTerm2);

J = J + regTerm;


% backpropagation
% Loop over all training examples
  % Step 1
  % Set the input layer’s values (a (1) ) to the t-th training example x (t) .
  % Perform a feedforward pass, computing the activations (z (2) , a (2) , z (3) , a (3) )
  % for layers 2 and 3. Note that you need to add a +1 term to ensure that
  % the vectors of activations for layers a (1) and a (2) also include the bias unit
  % This is done previously

for t = 1:m
  % Step 2. Select the delta for each k output unit
  for k = 1:num_labels
      yk = y(t) == k;
      delta_3(k) = h(t, k) - yk; % Get the k element in h and y
  end
  % Step 3. Compute delta for layer 2
  z2ExampleT = z2(t, :);
  delta_2 = Theta2' * delta_3' .* sigmoidGradient([1, z2ExampleT])';
  % Remove delta(0)
  delta_2 = delta_2(2:end);

  % Step 4. Accumulate
  Theta1_grad = Theta1_grad + delta_2 * a1(t, :);
  Theta2_grad = Theta2_grad + delta_3' * a2(t, :);
end

% Obtain the (unregularized) gradient for the neural network cost func-
% tion by dividing the accumulated gradients by m 1 :
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Regularize gradient
% To account for regularization, it
% turns out that you can add this as an additional term after computing the
% gradients using backpropagation.
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
