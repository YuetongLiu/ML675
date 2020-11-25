""" 
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np


class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures

    def fit(self, *, X, y, lr):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            lr: A float, the learning rate of this fit step.
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class Perceptron(Model):

    def __init__(self, *, nfeatures):
        super().__init__(nfeatures=nfeatures)
        self.W = np.zeros((nfeatures, 1))

    def fit(self, *, X, y, lr):
        """ TODO: Implement this!

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            lr: A float, the learning rate of this fit step.
        """
        n, d = X.shape

        X = X.todense()

        for i in range(n):
            x_p = X[i]
            y_p = 1 if y[i] == 1 else -1

            y_hat = np.dot(x_p, self.W)

            y_hat = 1 if y_hat > 0 else -1

            self.W += lr *  (y_p - y_hat) * x_p.T
        '''
        y_trans = y.reshape(len(y), 1)
        

        #X = X.todense()
        # y_hat = np.matmul(X, self.W)

        
        for idx in range(len(y_hat)):
                y_hat[idx] = 1 if y_hat[idx] > 0 else -1
                y_trans[idx] = 1 if y_trans[idx] == 1 else -1

        
        self.W += lr * X.T * (y_trans - y_hat)
        # print(y)
        # raise Exception("You must implement this method!")'''
        return

    def predict(self, X):
        """ TODO: Implement this!

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        X = self._fix_test_feats(X)

        X = X.todense()

        y_hat = np.matmul(X, self.W)

        for idx in range(len(y_hat)):
                y_hat[idx] = 1 if y_hat[idx] >= 0 else 0

        y_hat = np.squeeze(np.asarray(y_hat))

        y_hat = y_hat.astype(int)

        return y_hat

        #raise Exception("You must implement this method!")


class LogisticRegression(Model):

    def __init__(self, *, nfeatures):
        super().__init__(nfeatures=nfeatures)
        self.W = np.zeros((nfeatures, 1))

    '''
    def gradient(self, x, y):
        dw = np.zeros(self.W.shape)
        for j in range(dw.shape[0]):
            dw[j] = y * self.sigmoid(-np.dot(self.W.reshape(-1), x)) * x[j] \
                        + (1-y) * self.sigmoid(np.dot(self.W.reshape(-1), x)) * (-x[j])
        return dw
    '''



    def fit(self, *, X, y, lr):
        """ TODO: Implement this!

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            lr: A float, the learning rate of this fit step.
        """
        n, d = X.shape

        X = X.todense()
        '''

        for i in range(n):
            x_p = X[i].reshape((self.num_input_features,1))
            y_p = y[i]

            dw = np.zeros(self.W.shape)
            for j in range(dw.shape[0]):
                dw[j] = y_p * self.sigmoid(-np.dot(self.W.reshape(-1), x_p)) * x_p[j] \
                        + (1-y_p) * self.sigmoid(np.dot(self.W.reshape(-1), x_p)) * (-x_p[j])

            # dw = self.gradient(x_p, y_p)
            self.W += lr * dw

        return
        '''    
        for i in range(n):
            x_p = X[i]
            y_p = y[i]

            logits = np.dot(x_p, self.W)
            h = self.sigmoid(logits) 


            gradient = np.dot(x_p.T, (h - y_p))

        #error = yHat - y   
        #gradient = np.dot(X.T, error)
        #a = gradient.sum(axis=1, dtype='float') 

            self.W -= lr * gradient
        #raise Exception("You must implement this method!")

    def predict(self, X):
        """ TODO: Implement this!

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        X = self._fix_test_feats(X)

        X = X.todense()
        logits = np.dot(X, self.W)
        y_hat = self.sigmoid(logits) 

        for idx in range(len(y_hat)):
                y_hat[idx] = 1 if y_hat[idx] > 0.5 else 0

        y_hat = np.squeeze(np.asarray(y_hat))

        y_hat = y_hat.astype(int)

        #print(y_hat)
        return y_hat
        #raise Exception("You must implement this method!")

    def sigmoid(self, logits):
        """ TODO: Implement this! Write the sigmoid function
        Args:
            logits: array of log-odds for Logistic Regression

        Returns:
            An array of the sigmoid function being applied to the input
        """
        out = np.zeros((logits.shape[0], 1))

        for i in range(logits.shape[0]):

            if logits[i] > 0:
                out[i] = 1/(1 + np.exp(-logits[i]))
            else: 
                out[i] = np.exp(logits[i])/(1 + np.exp(logits[i]))
        
        return out
        #raise Exception("You must implement this method!")

