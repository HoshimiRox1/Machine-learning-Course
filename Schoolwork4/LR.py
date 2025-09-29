# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cost_gradient(W, X, Y, n):
      G = ###### Gradient
      j = ###### cost with respect to current W
      
    return (j, G)

def gradientDescent(W, X, Y, n, lr, iterations):
      J = np.zeros([iterations, 1])
      
      for i in range(iterations):
          (J[i], G) = cost_gradient(W, X, Y, n)
          W = ###### Update W based on gradient

      return (W,J)

def error(W, X, Y):
    Y_hat = np.sign(X@W)
    Y_hat = np.maximum(Y_hat, 0)
    
    return (1 - np.mean(np.equal(Y_hat, Y)))

iterations = ###### Training loops
lr = ###### Learning rate

data = np.loadtxt('LR.txt', delimiter=',')

n = data.shape[0]
W = np.random.random([3, 1])
X = np.concatenate([np.ones([n, 1]), data[:,0:2]], axis=1)
Y = np.expand_dims(data[:, 2], axis=1)

(W,J) = gradientDescent(W, X, Y, n, lr, iterations)
print(error(W, X, Y))

#Draw figure
idx0 = (data[:, 2]==0)
idx1 = (data[:, 2]==1)

plt.figure()
plt.ylim(-12,12)
plt.plot(data[idx0,0], data[idx0,1],'go')
plt.plot(data[idx1,0], data[idx1,1],'rx')

x1 = np.arange(-10,10,0.2)
y1 = (W[0] + W[1]*x1) / -W[2]
plt.plot(x1, y1)

plt.figure()
plt.plot(range(iterations), J)