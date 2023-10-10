import numpy as np
import pandas as pd


class LogisticRegression(object):

    '''
    Logistic Regression Classifier, with gradient descent cal

    You may regard each of this new LR model as a blank key fob
    After you add X and y into the model and fit it, it becomes a model to predict certain result
    just like an actual fob to open certain door

    Parameters:
        eta: float
            learning rate in (0.0, 1.0)
        n_iter: int
            the maximum number of iterations or epochs before stopping
            may stop the algorithm before convergence
            passes over the training dataset
        random_state: int
            random number generator seed for ramdom weight initialization
            may used to promise a reproducible result

    Attributes:
        w__: 1d-array
            wights after fitting
            w[0] denotes the intercept of the function, while each following denotes a slope of a dimension
        cost__: list
            sum of squares cost function value in each epoch
    '''



    # the initialize of the model

    def __init__(self, eta = 0.05, n_iter = 100, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state



    # tools for fit

    def net_input(self, X):
        '''
        to predict the result based on the current weight
        '''
        return np.dot(X, self.w__[1: ]) + self.w__[0]
    
    def phi(self, z):
        '''
        compute the logistic sigmoid activation
        '''
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))



    # fit itself

    def fit(self, X, y):

        '''
        fit training data

        Parameters:
            X: {array like}, shape = [n_samples, n_features]
                training vectors
                n_samples: number of the observations
                n_features: number of features
                
            y: array like, shape = [n_samples]
                the target values for supervised learning

        Returns:
            self: object
        '''

        rgen = np.random.RandomState(self.random_state)

        self.w__ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost__ = []

        for i in range(self.n_iter):

            # the logit(p) and p
            z = self.net_input(X)
            # phi_z is the probability of y[i] = 1
            phi_z = self.phi(z)

            # calculate the errors and modify the weight based on the result and learning rate
            errors = (y - phi_z)
            self.w__[1: ] += self.eta * np.dot(X.T, errors)
            self.w__[0] += self.eta * errors.sum()

            # the logistic cost
            cost = np.dot(-y, np.log(phi_z)) - np.dot((1 - y), np.log(1 - phi_z))
            self.cost__.append(cost)

        return self
    


    # the prediction

    def predict(self, X):
        '''
        return class label after unit step
        '''
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    


    # test model imported successfully
    def test_import(self):
        print("Model created!")
        print("eta = " + str(self.eta))
        print("n_iter = " + str(self.n_iter))
        print("ramdom_state = " + str(self.random_state))