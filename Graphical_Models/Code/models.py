"""
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np
np.seterr(all='raise') # Check numerical instability issues
import random
from scipy.special import digamma
from tqdm import tqdm
import scipy.sparse as sparse

class LDA(object):
    def __init__(self, *, inference):
        self.inference = inference
        self.topic_words = None


    def fit(self, *, X, iterations, estep_iterations):
        self.inference.inference(X=X, iterations=iterations, estep_iterations=estep_iterations)


    def predict(self, *, vocab, K):
        self.topic_words = {}
        preds = []
        for i, topic_dist in enumerate(self.inference.phi):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(K+1):-1]
            self.topic_words[i] = topic_words.tolist()
            preds.append('Topic {}: {}'.format(i, ' '.join(topic_words)))
        return preds


class Inference(object):
    """ Abstract inference object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, num_topics, num_docs, num_words, alpha, beta):
        self.num_topics = num_topics
        self.num_docs = num_docs
        self.num_words = num_words
        self.alpha = alpha
        self.beta = beta
        self.theta = np.zeros((num_docs, num_topics))
        self.phi = np.zeros((num_topics, num_words))

    def inference(self, *, X, iterations):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        raise NotImplementedError()


class MeanFieldVariationalInference(Inference):

    def __init__(self, *, num_topics, num_docs, num_words, alpha, beta, epsilon):
        super().__init__(num_topics, num_docs, num_words, alpha, beta)
        self.pi = np.zeros((self.num_docs, self.num_words, self.num_topics))
        self.gamma = np.zeros((self.num_docs, self.num_topics))
        self.lmbda = np.zeros((self.num_topics, self.num_words))
        self.sufficient_statistics = np.zeros((self.num_topics, self.num_words)) # sufficient statistics
        self.epsilon = epsilon

    def initialize_lambda(self):
        np.random.seed(0)
        self.lmbda = np.random.gamma(100, 1/100, (self.num_topics, self.num_words))

    def initialize_gamma(self):
        np.random.seed(0)
        self.gamma = np.random.gamma(100, 1/100, (self.num_docs, self.num_topics))

    def inference(self, *, X, iterations, estep_iterations):
        """
        Perform Mean Field Variational Inference using EM.
        Note: labels are not used here.

        You can use tqdm using the following:
            for t in tqdm(range(iterations))

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of EM iterations
            estep_iterations: int giving max number of E-step iterations
        """
        # TODO: Implement this!

        self.initialize_lambda()    

        E_phi = digamma(self.lmbda) - np.reshape(digamma(np.sum(self.lmbda,axis=1)),(-1,1))

        for iterate in range(iterations):

            # E-steps
            # 1. Initialize variational parameter 
            self.initialize_gamma()

            self.sufficient_statistics = np.zeros((self.num_topics, self.num_words))


            # 2. For each document d = 1...D
            for d in range(self.num_docs):
                
                # (a) For each iteration m = 1...M
                for m in range(estep_iterations):

                    
                    # i. Compute Eq[log(theta_dk)] based on most recent gamma_dk
                    E_theta = digamma(self.gamma[d,:]) - digamma(np.sum(self.gamma[d,:]))
                    
                    # ii. Compute pi_dik based on current document d:
                    # 
                    x_d = X.col[np.where(X.row == d)]
                    E_phi_d = E_phi[:,x_d] 

                    self.pi[d,x_d,:] = (np.exp(np.reshape(E_theta,(-1,1))) * np.exp(E_phi_d)).T

                    # iii. Update variational parameter d (for document-topic portions theta_d)
                    nor = np.sum(self.pi[d,x_d,:], axis = 1)
                    pi_sum = (self.pi[d,x_d,:]/np.reshape(nor,(-1,1)))

                    xd = X.data[np.where(X.row == d)]

                    gamma = self.alpha + xd @ pi_sum

                    #iv. If break from loop early (change in dk is smaller than epsilon).
                    diff = np.abs(gamma - self.gamma[d,:])
                    self.gamma[d,:] = gamma


                    if (1/self.num_topics)*np.sum(diff) < self.epsilon:
                                                break


                # 3. Compute suffcient statistics S(lambda)
                self.sufficient_statistics[:,x_d] += np.outer(np.exp(E_theta), xd/nor) * np.exp(E_phi_d)
           
            # M-steps
            # 1. Update variational parameter lambda_k
            self.lmbda = self.beta + self.sufficient_statistics

            # 2. Compute Eq[log(theta_kw] based on most recent lambda_kw
            lmbda_sum = np.sum(self.lmbda,axis=1)

            E_phi = digamma(self.lmbda) - np.reshape(digamma(lmbda_sum),(-1,1))
        
        # Update theta and phi
        self.theta = self.gamma/self.gamma.sum(axis = 0)  

        self.phi = self.lmbda/np.reshape(self.lmbda.sum(axis = 1),(-1,1))