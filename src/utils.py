# src/utils.py
from typing import Optional
import numpy as np

class GMA(object):
    """
    Gaussian Multiplicative Approximation of two variables in a vector
    Input:
    mu: mean vector of all the variables
    var: full covariance matrix of all the variables
    """

    def __init__(
            self, 
            mu: np.ndarray,
            var: np.ndarray,
            index1: Optional[int] = None,
            index2: Optional[int] = None,
            replace_index: Optional[int] = None,
            )-> None:
        
        self.mu = mu
        self.var = var
        if index1 is not None and index2 is not None and replace_index is not None:
            self.multiply_and_augment(index1, index2)
            self.swap(-1, replace_index)
            self.delete(-1)
    
    def multiply_and_augment(self, index1, index2):   
        "The multiplication of two variables is augmented to the last index of the provided vector"
        # Augment the dimension of the input matrix
        GMA_mu = np.vstack((self.mu, 0))
        GMA_var = np.append(self.var, np.zeros((1, self.var.shape[1])), axis=0)
        GMA_var = np.append(GMA_var, np.zeros((GMA_var.shape[0], 1)), axis=1)
        
        # Multiply the two provided indices
        # # Mean for the multiplicated term
        GMA_mu[-1] = self.mu[index1] * self.mu[index2] + self.var[index1][index2]
        # # Variance for the multiplicated term
        GMA_var[-1, -1] = self.var[index1][index1] * self.var[index2][index2] + self.var[index1][index2] ** 2 +\
                        2 * self.mu[index1] * self.mu[index2] * self.var[index1][index2] +\
                        self.var[index1][index1] * self.mu[index2] ** 2 + self.var[index2][index2] * self.mu[index1] ** 2
        # # Covariance between the multiplicated term and the existing terms
        for i in range(len(self.mu)):
            cov_i = self.var[i][index1] * self.mu[index2] + self.var[i][index2] * self.mu[index1]
            GMA_var[i][-1] = cov_i
            GMA_var[-1][i] = cov_i

        self.mu = GMA_mu
        self.var = GMA_var
        pass
    
    def swap(self, index1, index2):
        "Swap the sequence of moments of two variables in the vector"
        self.mu[[index1, index2]] = self.mu[[index2, index1]]
        self.var[[index1, index2]] = self.var[[index2, index1]]
        self.var[:, [index1, index2]] = self.var[:, [index2, index1]]
        pass
    
    def delete(self, index):
        "Delete the moments of a variables in the vector"
        self.mu = np.delete(self.mu, index, axis=0)
        self.var = np.delete(self.var, index, axis=0)
        self.var = np.delete(self.var, index, axis=1)
        pass

    def get_results(self):
        return self.mu, self.var