# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')

    n = data.shape[0]
    
    ###### You may modify this section to change the model
    X = np.concatenate([np.ones([n, 1]),
                    data[:, 0:8],  # 8个原始特征 (1次方)
                    np.power(data[:, 0:8], 2) # 8个特征的平方项 (2次方)
                   ], axis=1)
    ###### You may modify this section to change the model

    Y = None
    if "train" in addr:
        Y = np.expand_dims(data[:, -1], axis=1)
    
    return (X,Y,n)

def cost_gradient(W, X, Y, n, lambd):
    Y_hat = 1 / (1 + np.exp(-X @ W))
    
    loss = (-1 / n) * np.sum((Y * np.log(Y_hat) + (1-Y) * np.log(1- Y_hat + np.spacing(1))) )
    regu = (lambd / 2) * np.sum(W * W)
    
    G = (1 / n) * (X.T @ (Y_hat - Y)) + lambd * W
    j = loss + regu
    
    return (j, G)

def train(W, X, Y, lr, n, iterations, lambd):
    J = np.zeros([iterations, 1])
    
    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n, lambd)
        W = W - lr*G
    err = error(W, X, Y)
    
    return (W,J,err)

def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X@W))
    Y_hat[Y_hat<0.5] = 0
    Y_hat[Y_hat>0.5] = 1
    
    return (1-np.mean(np.equal(Y_hat, Y)))

def predict(W):
    (X, _, _) = read_data("Schoolwork6/test_data.csv")
    
    Y_hat = 1 / (1 + np.exp(-X@W))
    Y_hat[Y_hat<0.5] = 0
    Y_hat[Y_hat>0.5] = 1
    
    idx = np.expand_dims(np.arange(1,201), axis=1)
    np.savetxt("Schoolwork6/predict.csv", np.concatenate([idx, Y_hat], axis=1), header = "Index,ID", comments='', delimiter=',')
    
iterations = 1000
lr = 0.05
lambd = 0.035

(X, Y, n) = read_data("Schoolwork6/train.csv")
W = np.random.random([X.shape[1], 1])

(W,J,err) = train(W, X, Y, lr, n, iterations, lambd)
print(err)

plt.figure()
plt.plot(range(iterations), J)
plt.show()

predict(W)