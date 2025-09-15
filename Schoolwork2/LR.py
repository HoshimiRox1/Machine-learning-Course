# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cost_gradient(W, X, Y, n):
      ###### Gradient
      h = np.dot(X, W)
      diff = h - Y
      G = (1 / n) * np.dot(X.T, diff)
      ###### cost with respect to current W
      j = (1/(2*n)) * np.sum(np.square(diff))
      
      return (j, G)

def gradientDescent(W, X, Y, lr, iterations):
      n = np.size(Y)
      J = np.zeros([iterations, 1])
      
      for i in range(iterations):
          (J[i], G) = cost_gradient(W, X, Y, n)
          ###### Update W based on gradient
          W = W - G * lr

      return (W,J)

iterations = 1000
lr = 0.0001

data = np.loadtxt("Schoolwork2/LR.txt", delimiter=',')

n = np.size(data[:, 1])
W = np.zeros([2, 1])
X = np.c_[np.ones([n, 1]), data[:,0]]
Y = data[:, 1].reshape([n, 1])

(W,J) = gradientDescent(W, X, Y, lr, iterations)

#Draw figure
plt.figure()
plt.plot(data[:,0], data[:,1],'rx')
plt.plot(data[:,0], np.dot(X,W))

plt.figure()
plt.plot(range(iterations), J)
plt.show()