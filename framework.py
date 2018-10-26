import numpy as np
from helpers import *

class NN:
    def __init__(self, data, target, architecture):
        self.X = data
        [_, h, _] = architecture
        (n, m) = self.X.shape


        self.h = h
        self.n = n
        self.m = m
        self.Y = target.reshape((1, m))

        self.W1 = 0.01 * np.random.randn(h, n)
        self.b1 = np.zeros((h, 1))

        self.W2 = 0.01 * np.random.randn(1, h)
        self.b2 = 0
    
    def forward(self, X):
        W1 = self.W1
        b1 = self.b1
        W2 = self.W2
        b2 = self.b2
        
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        
        cache = {"Z1": Z1,
                "A1": A1,
                "Z2": Z2,
                "A2": A2}
        
        return A2, cache
    
    def cost(self, A2):
        m = self.m

        assert A2.shape == self.Y.shape

        logprobs = (np.multiply(self.Y, np.log(A2))) + (np.multiply(1-self.Y, np.log(1-A2)))
        cost = (-1/m) * logprobs.sum()
        cost = np.squeeze(cost)

        assert isinstance(cost, float)

        return cost
    
    def backward(self, cache):
        m = self.m
        
        W2 = self.W2
        
        A1 = cache['A1']
        A2 = cache['A2']
        
        dZ2 = A2-self.Y
        dW2 = (1/m)*np.dot(dZ2, A1.T)
        db2 = (1/m)*dZ2.sum(axis=1, keepdims=True)
        dZ1 = np.multiply((np.dot(W2.T, dZ2)), (1-np.square(A1)))
        dW1 = (1/m)*np.dot(dZ1, self.X.T)
        db1 = (1/m)*dZ1.sum(axis=1, keepdims=True)
        
        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
    
        self.W1 = self.W1 - learning_rate*dW1
        self.b1 = self.b1 - learning_rate*db1
        self.W2 = self.W2 - learning_rate*dW2
        self.b2 = self.b2 - learning_rate*db2

    def train(self, num_iterations = 1000, learning_rate=0.0001, print_cost=True):
        
        for i in range(0, num_iterations):
            A2, cache = self.forward(self.X)
            
            cost = self.cost(A2)

            grads = self.backward(cache)
    
            self.update_parameters(grads, learning_rate)
            
            ### END CODE HERE ###
            
            # Print the cost every 100 iterations
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

    # GRADED FUNCTION: predict

    def predict(self, X):
        A2, _ = self.forward(X)
        A2 = A2.round()
        return A2