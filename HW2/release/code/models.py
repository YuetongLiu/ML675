""" 
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np
import math

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

    def fit(self, X, y, lr=1.0, number_of_iterations=5, size=1):
        """ TODO: Implement this!

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            lr: A float, the learning rate of this fit step.
        """
        self.num_input_features = X.shape[1]

        # Initialize w to 0
        w = np.zeros((np.size(X, 1), 1))
        if size is None:
            size = np.size(X, 0)

        for idx in range(number_of_iterations):
            k = int(math.ceil(np.size(X, 0) / size))
            
            for i in range(k - 1):
                x_row = X[i * size:(i + 1) * size]
                y_row = y[i * size:(i + 1) * size]
                # print(x_row)
                # print(y_row)
                # print(w)

                # Calculate (w dot xi)
                y_hat = np.matmul(x_row, w)
                # print('Dot Prod:', y_hat)

                # Apply y^ = sign(w dot xi)
                for idx in range(len(y_hat)):
                    y_hat[idx] = 1 if y_hat[idx] >= 0 else -1
                
                # print('y_hat', y_hat)
                # print('y_row', y_row)
                
                # If y^ != yi, make an update to w.
                # w' = w + (lr * yi * xi) 
                productxiyi = np.zeros((np.size(x_row, 1), 1))
                if y_hat != y_row:
                    productxiyi = x_row.T * y_row
                    w = w + np.multiply(lr, productxiyi)

        self.weights = w
        print(self.weights)
        print('------------------------------')

        pass
        # raise Exception("You must implement this method!")

    def predict(self, X):
        """ TODO: Implement this!

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        X = self._fix_test_feats(X)
        raise Exception("You must implement this method!")


class LogisticRegression(Model):

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

        self.w = np.zeros(d)

        H = self.sigmoid(np.dot(w.T,X)) 

        dw = np.dot(X,(H-Y).T)/d

        raise Exception("You must implement this method!")

    def predict(self, X):
        """ TODO: Implement this!

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        X = self._fix_test_feats(X)
        return np.sign(X@self.w)

        #raise Exception("You must implement this method!")



    def sigmoid(self, logits):
        """ TODO: Implement this! Write the sigmoid function

        Args:
            logits: array of log-odds for Logistic Regression

        Returns:
            An array of the sigmoid function being applied to the input
        """
        s = 1/(1 + np.exp(-logits)) 
        return s

        # raise Exception("You must implement this method!")

