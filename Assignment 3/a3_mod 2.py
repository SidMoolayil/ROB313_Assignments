import matplotlib.pyplot as plt
import numpy as np
# add the data folder to path (this specifies a relative path to where my data is stored)
import os, sys
sys.path.append(os.path.join(os.getcwd(), "../../../data"))
from data_utils import load_dataset
import autograd.numpy as np
from autograd import value_and_grad

def update_parameters(w, grad_w, learning_rate=1.):
    """
    perform gradient descent update to minimize an objective
    Inputs:
        w : vector of parameters
        grad_w : gradient of the loss with respect to the parameters
        learning_rate : learning rate of the optimizer
    """
    return w - learning_rate * grad_w

def forward_pass(W1, W2, W3, b1, b2, b3, x):
    """
    forward-pass for an fully connected neural network with 2 hidden layers of M neurons
    Inputs:
        W1 : (M, 784) weights of first (hidden) layer
        W2 : (M, M) weights of second (hidden) layer
        W3 : (10, M) weights of third (output) layer
        b1 : (M, 1) biases of first (hidden) layer
        b2 : (M, 1) biases of second (hidden) layer
        b3 : (10, 1) biases of third (output) layer
        x : (N, 784) training inputs
    Outputs:
        Fhat : (N, 10) output of the neural network at training inputs
    """
    H1 = np.maximum(0, np.dot(x, W1.T) + b1.T) # layer 1 neurons with ReLU activation, shape (N, M)
    H2 = np.maximum(0, np.dot(H1, W2.T) + b2.T) # layer 2 neurons with ReLU activation, shape (N, M)
    Fhat = np.dot(H2, W3.T) + b3.T # layer 3 (output) neurons with linear activation, shape (N, 10)
    # ORIGINAL CODE
    """
    return Fhat
    """
    # MODIFIED CODE
    """"""
    Fhatmax = Fhat.max(axis=1, keepdims=True)
    return Fhat - (Fhatmax + np.log(np.sum(np.exp(Fhat - Fhatmax), axis=1, keepdims=True)))
    """"""

def negative_log_likelihood(W1, W2, W3, b1, b2, b3, x, y):
    """
    computes the negative log likelihood of the model `forward_pass`
    Inputs:
        W1, W2, W3, b1, b2, b3, x : same as `forward_pass`
        y : (N, 10) training responses
    Outputs:
        nll : negative log likelihood
    """
    Fhat = forward_pass(W1, W2, W3, b1, b2, b3, x)
    # ORIGINAL CODE
    """
    nll = 0.5*np.sum(np.square(Fhat - y)) + 0.5*y.size*np.log(2.*np.pi)     
    """
    # MODIFIED CODE (assuming `Fhat` are the class conditional log probabilities)
    """"""
    nll = -np.sum(Fhat[y])
    """"""
    return nll

nll_gradients = value_and_grad(negative_log_likelihood, argnum=[0,1,2,3,4,5])
"""
    returns the output of `negative_log_likelihood` as well as the gradient of the 
    output with respect to all weights and biases
    Inputs:
        same as negative_log_likelihood (W1, W2, W3, b1, b2, b3, x, y)
    Outputs: (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad))
        nll : output of `negative_log_likelihood`
        W1_grad : (M, 784) gradient of the nll with respect to the weights of first (hidden) layer
        W2_grad : (M, M) gradient of the nll with respect to the weights of second (hidden) layer
        W3_grad : (10, M) gradient of the nll with respect to the weights of third (output) layer
        b1_grad : (M, 1) gradient of the nll with respect to the biases of first (hidden) layer
        b2_grad : (M, 1) gradient of the nll with respect to the biases of second (hidden) layer
        b3_grad : (10, 1) gradient of the nll with respect to the biases of third (output) layer
     """;



np.random.seed(1)

# load the MNIST_small dataset
from data_utils import load_dataset
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')

# specify optimization parameters for each neural network model
learn_rate = 0.0005
max_iter = 5000
n_hidden = 100 # specify number of neurons per hidden layer
print("Training network with %d hidden units per layer."%n_hidden)
# initialize the weights and biases of the network
W1 = np.random.randn(n_hidden, 784) # (unscaled) weights of first (hidden) layer
W1 /= np.sqrt(0.5*W1.shape[1]) # scale the weights using Xavier initialization (for ReLU)
W2 = np.random.randn(n_hidden, n_hidden) # (unscaled) weights of second (hidden) layer
W2 /= np.sqrt(0.5*W2.shape[1]) # scale the weights using Xavier initialization (for ReLU)
W3 = np.random.randn(10, n_hidden) # (unscaled) weights of third (output) layer
W3 /= np.sqrt(W3.shape[1]) # scale the weights using Xavier initialization (not for ReLU)
b1 = np.zeros((n_hidden, 1)) # biases of first (hidden) layer
b2 = np.zeros((n_hidden, 1)) # biases of second (hidden) layer
b3 = np.zeros((10, 1)) # biases of third (output) layer

# begin training iterations
best = dict(valid_nll=np.inf) # dictionary to store network with minimum validation loss
train_nll_curve, valid_nll_curve, iter_curve = [], [], [] # values to save for plotting
iteration = 0
while iteration <= max_iter:
    epoch_order = np.random.permutation(x_train.shape[0]) # shuffle the training set
    for minibatch in epoch_order.reshape((-1, 250)):
        # compute the gradient
        (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = \
            nll_gradients(W1, W2, W3, b1, b2, b3, x_train[minibatch], y_train[minibatch])
        # save training and validation nll
        if iteration==0 or (iteration % (10000//200)) == 0:
            valid_nll = negative_log_likelihood(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
            train_nll_curve.append(nll); valid_nll_curve.append(valid_nll); iter_curve.append(iteration)
            if iteration==0 or (iteration % 1000) == 0: # then print
                print("Iter %3d, train nll = %.6f, valid nll = %.6f" % (iteration, nll, valid_nll))
            if valid_nll < best["valid_nll"]: # then save this model
                best = dict(valid_nll=valid_nll, iteration=iteration,
                           parameters=[param.copy() for param in [W1, W2, W3, b1, b2, b3]])
        # update the parameters
        W1 = update_parameters(W1, W1_grad, learn_rate)
        W2 = update_parameters(W2, W2_grad, learn_rate)
        W3 = update_parameters(W3, W3_grad, learn_rate)
        b1 = update_parameters(b1, b1_grad, learn_rate)
        b2 = update_parameters(b2, b2_grad, learn_rate)
        b3 = update_parameters(b3, b3_grad, learn_rate)
        iteration += 1 # increment iteration counter
print("Done Training. Best model is from iteration %d with validation nll=%.6f" % \
      (best["iteration"], best["valid_nll"]))
[W1, W2, W3, b1, b2, b3] = best["parameters"] # get best model from early stopping
y_valid_pred = forward_pass(W1, W2, W3, b1, b2, b3, x_valid)
valid_accuracy = np.mean(np.argmax(y_valid_pred, axis=1) == np.argmax(y_valid, axis=1))
print("validation accuracy with this model = %.6f" % valid_accuracy)

# plot the training nll and validation curves
plt.figure(figsize=(8,8))
plt.plot(iter_curve, np.array(train_nll_curve)/250*x_train.shape[0], 'r', label="Training set NLL")
plt.plot(iter_curve, np.array(valid_nll_curve), 'b', label="Validation set NLL")
plt.legend(loc=0)
plt.xlabel("Iteration")
plt.ylabel("Negative Log Likelihood (NLL)")
plt.figure(figsize=(8,8))
plt.plot(iter_curve, np.array(train_nll_curve)/250, 'r', label="Training average NLL")
plt.plot(iter_curve, np.array(valid_nll_curve)/x_valid.shape[0], 'b', label="Validation average NLL")
plt.legend(loc=0)
plt.xlabel("Iteration")
plt.ylabel("Average Negative Log Likelihood (NLL)")

# evaluate the model on the test set
y_test_pred = forward_pass(W1, W2, W3, b1, b2, b3, x_test)
test_nll = negative_log_likelihood(W1, W2, W3, b1, b2, b3, x_test, y_test)
print("Test nll: %.6f" % test_nll)
test_accuracy = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test, axis=1))
print("Test accuracy: %.6f" % test_accuracy)
