""" 
Keep kernel implementations in here.
"""

import numpy as np
from collections import defaultdict, Counter
from functools import wraps
from tqdm import tqdm



def cache_decorator():
    """
    Cache decorator. Stores elements to avoid repeated computations.
    For more details see: https://stackoverflow.com/questions/36684319/decorator-for-a-class-method-that-caches-return-value-after-first-access
    """
    def wrapper(function):
        """
        Return element if in cache. Otherwise compute and store.
        """
        cache = {}

        @wraps(function)
        def element(*args):
            if args in cache:
                result = cache[args]
            else:
                result = function(*args)
                cache[args] = result
            return result

        def clear():
            """
            Clear cache.
            """
            cache.clear()

        # Clear the cache
        element.clear = clear
        return element
    return wrapper


class Kernel(object):
    """ Abstract kernel object.
    """
    def evaluate(self, s, t):
        """
        Kernel function evaluation.

        Args:
            s: A string corresponding to a document.
            t: A string corresponding to a document.

        Returns:
            A float from evaluating K(s,t)
        """
        raise NotImplementedError()

    def compute_kernel_matrix(self, *, X, X_prime=None):
        """
        Compute kernel matrix. Index into kernel matrix to evaluate kernel function.

        Args:
            X: A list of strings, where each string corresponds to a document.
            X_prime: (during testing) A list of strings, where each string corresponds to a document.

        Returns:
            A compressed sparse row matrix of floats with each element representing
            one kernel function evaluation.
        """
        X_prime = X if not X_prime else X_prime
        kernel_matrix = np.zeros((len(X), len(X_prime)), dtype=np.float32)

        # TODO: Implement this!
        # raise Exception("You must implement this method!")
        for i in range(len(X)):
        	for j in range(len(X_prime)):
        		kernel_matrix[i, j] = self.evaluate(X[i], X_prime[j])

        return kernel_matrix




class NgramKernel(Kernel):
    def __init__(self, *, ngram_length):
        """
        Args:
            ngram_length: length to use for n-grams
        """
        self.ngram_length = ngram_length


    def generate_ngrams(self, doc):
        """
        Generate the n-grams for a document.

        Args:
            doc: A string corresponding to a document.

        Returns:
            Set of all distinct n-grams within the document.
        """
        # TODO: Implement this!
        grams = set()

        for i in range(len(doc)-self.ngram_length+1):
        	grams.add(doc[i:i+self.ngram_length])
        return grams

        #raise Exception("You must implement this method!")


    @cache_decorator()
    def evaluate(self, s, t):
        """
        n-gram kernel function evaluation.

        Args:
            s: A string corresponding to a document.
            t: A string corresponding to a document.

        Returns:
            A float from evaluating K(s,t)
        """
        # TODO: Implement this!
        s = self.generate_ngrams(doc = s)
        t = self.generate_ngrams(doc = t)
        if not s.union(t):
        	return float(1)
        else: 
        	return len(s.intersection(t))/len(s.union(t))

        # raise Exception("You must implement this method!")


class TFIDFKernel(Kernel):
    def __init__(self, *, X, X_prime=None):
        """
        Pre-compute tf-idf values for each (document, word) pair in dataset.

        Args:
            X: A list of strings, where each string corresponds to a document.
            X_prime: (during testing) A list of strings, where each string corresponds to a document.

        Sets:
            tfidf: You will use this in the evaluate function.
        """
        self.tfidf = self.compute_tfidf(X, X_prime)
        

    def compute_tf(self, doc):
        """
        Compute the tf for each word in a particular document.
        You may choose to use or not use this helper function.

        Args:
            doc: A string corresponding to a document.

        Returns:
            A data structure containing tf values.
        """
        # TODO: Implement this!
        #raise Exception("You must implement this method!")
        counts = dict()
        words = doc.split()

        # count
        for word in words:
        	if word in counts:
        		counts[word] += 1
        	else:
        		counts[word] = 1

        # frequency
        for key in counts.keys():
        	counts[key] = counts[key]/len(words)

        return counts


    def compute_df(self, X, vocab):
        """
        Compute the df for each word in the vocab.
        You may choose to use or not use this helper function.

        Args:
            X: A list of strings, where each string corresponds to a document.
            vocab: A set of distinct words that occur in the corpus.

        Returns:
            A data structure containing df values.
        """
        # TODO: Implement this!
        # raise Exception("You must implement this method!")
        counts = dict()


        for v in vocab:
        	counts[v] = 0

        	for x in X:
        		x = x.split()
        		if v in x:
        			counts[v] += 1

        return counts


    def compute_tfidf(self, X, X_prime):
        """
        Compute the tf-idf for each (document, word) pair in dataset.
        You will call the helper functions to compute term-frequency 
        and document-frequency here.

        Args:
            X: A list of strings, where each string corresponds to a document.
            X_prime: (during testing) A list of strings, where each string corresponds to a document.

        Returns:
            A data structure containing tf-idf values. You can represent this however you like.
            If you're having trouble, you may want to consider a dictionary keyed by 
            the tuple (document, word).
        """
        # Concatenate collections of documents during testing
        if X_prime:
            X = X + X_prime

        #dic = defaultdict(float)

        #for document in X:
       # 	vocab = document.split()
       # 	tf = self.compute_tf(doc = document)
       # 	df = self.compute_df(X=X, vocab = vocab)

        #	for word in vocab:
       # 		dic[(document, word)] = tf[word]*np.log(len(X)/(df[word]+1))

        dic = defaultdict(float)
        words = []
        N = len(X)
        for d in X:
        	words.extend(d.split(' '))
        vocab = set(words)
        df = self.compute_df(X, vocab)
        for d in X:
        	tf = self.compute_tf(d)
        	for w in tf.keys():
        		dic[(d,w)] = tf[w]*np.log(N/(df[w]+1))

        return dic


        # TODO: Implement this!
        #raise Exception("You must implement this method!") 


    @cache_decorator()
    def evaluate(self, s, t):
        """
        tf-idf kernel function evaluation.

        Args:
            s: A string corresponding to a document.
            t: A string corresponding to a document.

        Returns:
            A float from evaluating K(s,t)
        """
        # TODO: Implement this!
        #raise Exception("You must implement this method!")
        k = 0
        s1 = s.split()
        tf = self.compute_tf(doc = t)

        t = t.split()
        
        

        for w in s1:
        	if w in t:
        		freq = tf[w]
        		#print(self.tfidf[(s,w)])
        		k += freq*self.tfidf[(s,w)]


        return k


