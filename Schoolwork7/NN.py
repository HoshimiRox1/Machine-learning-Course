# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Utilities
def onehotEncoder(Y, ny):
      # Y must be a column vector or 1D array of class indices
      # ny is the number of classes
      return np.eye(ny)[Y.flatten()]

# Xavier Initialization
def initWeights(M):
      l = len(M)
      W = []
      B = []
      
      for i in range(1, l):
            # Xavier initialization (He initialization is often better for ReLU, 
            # but sticking to original template intent)
            W.append(np.random.randn(M[i-1], M[i]) * np.sqrt(2/M[i-1])) 
            B.append(np.zeros([1, M[i]]))
            
      return W, B

# Forward propagation
def networkForward(X, W, B):
      l = len(W)
      A = [None for i in range(l+1)]
      A[0] = X # A[0] is the input Xgit 
      
      ##### Calculate the output of every layer A[i], where i = 0, 1, 2, ..., l

      # Layer 1 (Hidden Layer - ReLU)
      Z1 = np.dot(A[0], W[0]) + B[0]
      A[1] = np.maximum(0, Z1) # ReLU activation
      
      # Layer 2 (Output Layer - Softmax)
      Z2 = np.dot(A[1], W[1]) + B[1]
      # Softmax activation (stabilized)
      exp_Z2 = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True)) 
      A[2] = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)
      
      # NOTE: For backprop, we'd typically save Z1 and Z2 as well, 
      # but we'll recalculate the ReLU derivative from A[1].

      return A
#--------------------------

# Backward propagation
def networkBackward(Y, A, W):
      l = len(W)
      m = Y.shape[0] # Number of samples
      dW = [None for i in range(l)]
      dB = [None for i in range(l)]
      
      ##### Calculate the partial derivatives of all w and b in each layer dW[i] and dB[i], where i = 0, 1, ..., l-1
      
      # Layer L (Output Layer - Softmax + Cross-Entropy)
      # dZ2 = (A[2] - Y) / m is the gradient of the loss with respect to Z2
      dZ2 = (A[2] - Y) / m
      
      # Gradients for W[1] and B[1]
      dW[1] = np.dot(A[1].T, dZ2)
      dB[1] = np.sum(dZ2, axis=0, keepdims=True)
      
      # Layer L-1 (Hidden Layer - ReLU)
      # dA1 = dZ2 . W[1]^T is the gradient of the loss w.r.t A[1]
      dA1 = np.dot(dZ2, W[1].T)
      # dZ1 = dA1 * ReLU_derivative(Z1). ReLU derivative is 1 where A[1] > 0, and 0 otherwise.
      dZ1 = dA1 * (A[1] > 0)
      
      # Gradients for W[0] and B[0]
      dW[0] = np.dot(A[0].T, dZ1)
      dB[0] = np.sum(dZ1, axis=0, keepdims=True)

      return dW, dB
#--------------------------

# Update weights by gradient descent
def updateWeights(W, B, dW, dB, lr):
      l = len(W)
      
      for i in range(l):
            W[i] -= lr * dW[i]
            B[i] -= lr * dB[i]
            
      return W, B
#--------------------------

# Cost function (Cross-Entropy Loss)
def cost(A_last, Y, W):
      m = Y.shape[0]
      # A_last is the output of the network (A[l])
      
      # Numerical stability: use np.clip to prevent log(0)
      epsilon = 1e-12 
      A_last = np.clip(A_last, epsilon, 1. - epsilon)
      
      # Cross-Entropy Loss
      c = -np.sum(Y * np.log(A_last)) / m
      
      return c
#--------------------------

def train(X, Y, M, lr, iterations):
      W, B = initWeights(M)
      costs = []
      
      for i in range(iterations):
            A = networkForward(X, W, B)
            c = cost(A[-1], Y, W)
            dW, dB = networkBackward(Y, A, W)
            W, B = updateWeights(W, B, dW, dB, lr)

            if i % 100 == 0:
                  print("Cost after iteration %i: %f" %(i, c))
                  costs.append(c)

      return W, B, costs

def predict(X, W, B, Y):
      Y_out = np.zeros([X.shape[0], Y.shape[1]])
      
      A = networkForward(X, W, B)
      idx = np.argmax(A[-1], axis=1)
      Y_out[range(Y.shape[0]),idx] = 1
      
      return Y_out

def test(Y, X, W, B):
      Y_out = predict(X, W, B, Y)
      # Calculate accuracy using element-wise product (Y_out*Y) and then sum
      acc = np.sum(Y_out*Y) / Y.shape[0]
      print("Training accuracy is: %f" %(acc))
      
      return acc

def output(X, W, B):
    A = networkForward(X, W, B)
    
    # Get the predicted class index (0-9 for MNIST)
    Y_hat = np.expand_dims(np.argmax(A[-1], axis=1), axis=1)
    idx = np.expand_dims(np.arange(Y_hat.shape[0]), axis=1)
    
    # Create the final output array [sample_index, prediction]
    output = np.hstack([idx, Y_hat])

    # Save to a file (for submission or verification)
    np.savetxt("submission.csv", output, delimiter=",", fmt='%i')
    
    return output

iterations = 10000
lr = 0.01

X = np.load("Schoolwork7/train_data.npy")
Y = np.load("Schoolwork7/train_label.npy")
(n, m) = X.shape
Y = onehotEncoder(Y, 10)

M = [784, 25, 10]
W, B, costs = train(X, Y, M, lr, iterations)

plt.figure()
plt.plot(range(len(costs)), costs)

X = np.load("Schoolwork7/train_data.npy")
Y = np.load("Schoolwork7/train_label.npy")
Y = onehotEncoder(Y, 10)
test(Y, X, W, B)

X = np.load("Schoolwork7/test_data.npy")
output(X, W, B)

plt.show()
