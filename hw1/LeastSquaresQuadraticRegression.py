'''
Test Cases
----------
# Same test as test_on_toy_data() defined below

>>> prng = np.random.RandomState(0)
>>> N = 100

>>> true_w_F = np.asarray([1.1, -2.2])
>>> true_b = 0.0
>>> x_NF = prng.randn(N, 2)
>>> x_NF[:,1] = x_NF[:,0]**2
>>> y_N = true_b + np.matmul(x_NF, true_w_F) + 0.03 * prng.randn(N)

>>> x_N = x_NF[:,0]
>>> quad_regr = LeastSquaresQuadraticRegressor()
>>> quad_regr.fit(x_N, y_N)

>>> yhat_N = quad_regr.predict(x_N)
>>> np.set_printoptions(precision=3, formatter={'float':lambda x: '% .3f' % x})
>>> print(quad_regr.w_F)
[ 1.101 -2.194]
>>> print(np.asarray([quad_regr.b]))
[-0.008]
'''

import numpy as np
from LeastSquaresLinearRegression import LeastSquaresLinearRegressor
# No other imports allowed!

class LeastSquaresQuadraticRegressor(LeastSquaresLinearRegressor):
    ''' A quadratic regression model with sklearn-like API

    Given a single feature, constructs a second quadratic feature and
    fits a parabola by solving the "least squares" optimization problem.

    Attributes
    ----------
    * self.w_F : 1D numpy array, size n_features (= 2)
        vector of weights, one value for each feature
    * self.b : float
        scalar real-valued bias or "intercept"
    '''

    def __init__(self):
        ''' Constructor of an sklearn-like regressor

        Should do nothing. Attributes are only set after calling 'fit'.
        '''
        # Leave this alone
        pass

    def fit(self, x_N, y_N):
        r''' Compute and store weights that solve least-squares problem.

        Args
        ----
        x_N : 1D numpy array, shape (n_examples,) = (N,)
            Input measurement ("feature") for all examples in train set.
        y_N : 1D numpy array, shape (n_examples,) = (N,)
            Response measurements for all examples in train set.

        Returns
        -------
        Nothing. 

        Post-Condition
        --------------
        Internal attributes updated:
        * self.w_F (vector of weights for linear and quadratic feature)
        * self.b (scalar real bias, if desired)

        Notes
        -----
        The least-squares optimization problem is:
        
        .. math:
            \min_{w \in \mathbb{R}^F, b \in \mathbb{R}}
                \sum_{n=1}^N (y_n - b - \sum_f x_{nf} w_f)^2
        '''      
        
        # Currently just calls your linear regression code.
        # You should be adding a quadratic feature
        return super().fit(x_N[:,np.newaxis], y_N) # TODO fixme


    def predict(self, x_M):
        ''' Make predictions given input features for M examples

        Args
        ----
        x_M : 1D numpy array, shape (n_examples,)
            Input measurements ("features") for all examples of interest.

        Returns
        -------
        yhat_M : 1D array, size M
            Each value is the predicted scalar for one example
        '''
        # Currently just calls your linear regression code.
        # You should be adding a quadratic feature
        return super().predict(x_M[:,np.newaxis]) # TODO fixme


def test_on_toy_data(N=100):
    '''
    Simple test case with toy dataset with N=100 examples
    created via a known linear regression model plus small noise.

    The test verifies that our LR can recover true w and b parameter values.
    '''
    prng = np.random.RandomState(0)

    true_w_F = np.asarray([1.1, -2.2])
    true_b = 0.0
    x_NF = prng.randn(N, 2)
    x_NF[:,1] = x_NF[:,0]**2
    y_N = true_b + np.matmul(x_NF, true_w_F) + 0.03 * prng.randn(N)

    x_N = x_NF[:,0]
    quad_regr = LeastSquaresQuadraticRegressor()
    quad_regr.fit(x_N, y_N)

    yhat_N = quad_regr.predict(x_N)

    np.set_printoptions(precision=3, formatter={'float':lambda x: '% .3f' % x})

    print("True weights")
    print(true_w_F)
    print("Estimated weights")
    print(quad_regr.w_F)

    print("True intercept")
    print(np.asarray([true_b]))
    print("Estimated intercept")
    print(np.asarray([quad_regr.b]))

if __name__ == '__main__':
    test_on_toy_data()
