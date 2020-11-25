""" 
Keep model implementations in here.
"""

import numpy as np


class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def fit(self, *, X, y, kernel_matrix):
        """ Fit the model.

        Args:
            X: A list of strings, where each string corresponds to a document.
            y: A dense array of ints with shape [num_examples].
            kernel_matrix: an ndarray containing kernel evaluations
        """
        raise NotImplementedError()

    def predict(self, X, kernel_matrix):
        """ Predict.

        Args:
            X: A list of strings, where each string corresponds to a document.
            kernel_matrix: an ndarray containing kernel evaluations

        Returns:s
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class KernelPegasos(Model):

    def __init__(self, *, nexamples, lmbda):
        """
        Args:
            nexamples: size of example space
            lmbda: regularizer term (lambda)

        Sets:
            b: beta vector (related to alpha in dual formulation)
            t: current iteration
            kernel_degree: polynomial degree for kernel function
            support_vectors: array of support vectors
            labels_corresp_to_svs: training labels that correspond with support vectors
        """
        super().__init__(lmbda=lmbda)
        self.b = np.zeros(nexamples, dtype=int)
        self.t = 1
        self.support_vectors = None
        self.labels_corresp_to_svs = None



    def fit(self, *, X, y, kernel_matrix):
        """ Fit the model.

        Args:
            X: A list of strings, where each string corresponds to a document.
            y: A dense array of ints with shape [num_examples].
            kernel_matrix: an ndarray containing kernel evaluations
        """
        # TODO: Implement this!
        #raise Exception("You must implement this method!")
        self.support_vectors = []
        self.labels_corresp_to_svs = []

        y_s = y
        for k in range(len(y)):
            if y[k] == 0:
                y_s[k] = -1

        for j in range(len(self.b)):
            self.t += 1
            val = 0
            for i in range(len(self.b)):
                val += self.b[i]*y_s[i]*kernel_matrix[i,j]


            req = y_s[j]/(self.lmbda*(self.t-1))*val
            if req < 1:
                self.b[j] += 1


            if self.b[j] > 0:
                self.support_vectors.append(X[j])
                self.labels_corresp_to_svs.append(y[j])







    def predict(self, *, X, kernel_matrix):
        """ Predict.

        Args:
            X: A list of strings, where each string corresponds to a document.
            kernel_matrix: an ndarray containing kernel evaluations

        Returns:
            A dense array of ints with shape [num_examples].
        """
        # TODO: Implement this!
        #raise Exception("You must implement this method!")
        a1 = [i for i in self.b if i != 0]
        
        a = np.divide(a1,(self.lmbda*self.t))
        y_bar = np.zeros(len(X), dtype=int)

        y_s = self.labels_corresp_to_svs

        for k in range(len(self.labels_corresp_to_svs)):
            if self.labels_corresp_to_svs[k] == 0:
                y_s[k] = -1

        
        for j in range(len(X)):
            val = 0
            for i in range(len(a)):
                val += a[i]*y_s[i]*kernel_matrix[i,j]

            if val > 0:
                y_bar[j] = 1 
            else:
                y_bar[j] = 0
        
        #print(kernel_matrix.shape)
        
        return y_bar



