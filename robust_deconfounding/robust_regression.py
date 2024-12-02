from typing import Set, Optional, Self
from numpy.typing import NDArray
import numpy as np
import scipy as sp
import statsmodels.api as sm
import itertools


class BaseRobustRegression:
    """Base class for robust regression models.

    Attributes:
        fit_intercept (bool): Whether to include an intercept in the model.
        model: Holds the regression model after fitting.
        inliers (list): Indices of inliers used in the final model.
    """

    def __init__(self, fit_intercept: bool = False):
        self.fit_intercept = fit_intercept
        self.model = None
        self.inliers = []
        self.coef=[]

    def fit(self, x: NDArray, y: NDArray) -> Self:
        """Fits the regression model to the data.

        Must be implemented by subclasses.

        Args:
            x (NDArray): Design matrix.
            y (NDArray): Response vector.

        Returns:
            Self: The instance itself.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    @staticmethod
    def _validate_inputs(x, y):
        """Validates the input data and basis."""
        if len(y) != len(x) or len(y) == 0:
            raise ValueError("Data and basis must have the same length and be non-empty.")

    @staticmethod
    def _add_intercept(x):
        """Adds an intercept column to the design matrix."""
        n = len(x)
        return np.hstack((np.ones((n, 1)), x))

    @property
    def coef_(self) -> NDArray:
        """Returns the coefficients of the regression model, excluding the intercept."""
        return self.coef

    @property
    def intercept_(self) -> Optional[float]:
        """Returns the intercept of the regression model, if applicable."""
        return self.model.params[0] if self.fit_intercept else None

    @property
    def inliers_(self) -> Set[int]:
        """Returns a set of indices of inliers used in the final model."""
        return set(self.inliers)


class Torrent(BaseRobustRegression):
    """Torrent algorithm for regression with robustness to outliers.

    Extends the base regression to implement an iterative process of fitting and refining inliers.

    Attributes:
        a (float): Proportion of data considered as inliers.
        max_iter (int): Maximum number of iterations.
        predicted_inliers (list): List to track inliers over iterations.

    Reference:
    Robust Regression via Hard Thresholding, Kush Bhatia, Prateek Jain, Purushottam Kar,
    https://arxiv.org/abs/1506.02428
    """

    def __init__(self, a: float, fit_intercept: bool = False, max_iter: int = 100):
        super().__init__(fit_intercept)
        if not 0 < a < 1:
            raise ValueError("'a' must be in the range (0, 1).")
        self.a = a
        self.max_iter = max_iter
        self.predicted_inliers = []

    def fit(self, x: NDArray, y: NDArray) -> Self:
        """Fit model using an iterative process to determine inliers and refit the model."""
        n = len(y)
        y = y.reshape(n, -1)

        self._validate_inputs(x, y)
        if self.fit_intercept:
            x = self._add_intercept(x)

        an = int(self.a * n)
        if an == 0:
            raise ValueError("'a' is too small. Increase 'a' or the number of data points .")

        self.inliers = list(range(n))
        self.predicted_inliers.append(self.inliers)

        for _ in range(self.max_iter):
            self.model = sm.OLS(y[self.inliers], x[self.inliers]).fit()

            err = np.linalg.norm(y - self.model.predict(x).reshape(n, -1), axis=1)

            old_inliers = self.inliers
            self.inliers = np.argpartition(err, an)[:an]
            self.predicted_inliers.append(self.inliers)

            if set(self.inliers) == set(old_inliers):
                break

        self.coef=self.model.params    
        return self


class BFS(BaseRobustRegression):
    """Brute Force Search (BFS) algorithm for regression to find the best subset of inliers.

    Attributes:
        a (float): Proportion of data to be considered for each potential subset of inliers.
    """

    def __init__(self, a: float, fit_intercept: bool = True):
        super().__init__(fit_intercept)
        if not 0 < a < 1:
            raise ValueError("a must be in the range (0, 1).")

        self.a = a

    def fit(self, x: NDArray, y: NDArray) -> Self:
        """Fit model by exhaustively searching over possible combinations of inliers."""
        n = y.shape[0]
        y = y.reshape(n, -1)

        self._validate_inputs(x, y)
        if self.fit_intercept:
            x = self._add_intercept(x)

        an = int(self.a * n)
        if an == 0:
            raise ValueError("'a' is too small. Increase 'a' or the number of data points.")

        permu = itertools.combinations(range(n), an)
        err_min = np.inf

        for p in permu:
            model = sm.OLS(y[list(p)], x[list(p)]).fit()

            err = np.linalg.norm(y[list(p)] - model.predict(x[list(p)]).reshape(an, -1))

            if err < err_min:
                self.inliers = list(p)
                self.model = model
                err_min = err

        self.coef=self.model.params  
        return self

class Torrent_reg(BaseRobustRegression):
    """Torrent algorithm for regression with robustness to outliers.

    Extends the base regression to implement an iterative process of fitting and refining inliers.

    Attributes:
        a (float): Proportion of data considered as inliers.
        max_iter (int): Maximum number of iterations.
        predicted_inliers (list): List to track inliers over iterations.
    """

    def __init__(self, a: float, fit_intercept: bool = False, max_iter: int = 100, K=np.array([0]), lmbd=0):
        super().__init__(fit_intercept)
        if not 0 < a < 1:
            raise ValueError("'a' must be in the range (0, 1).")
        self.a = a
        self.max_iter = max_iter
        self.predicted_inliers = []
        self.K=K
        self.lmbd=lmbd

    def fit(self, x: NDArray, y: NDArray) -> Self:
        """Fit model using an iterative process to determine inliers and refit the model.
            lambda: the regularization parameter
            K:      positive semi-definite matrix for the penalty
            """

        n = len(y)
        y = y.reshape(n, -1)

        self._validate_inputs(x, y)
        if self.fit_intercept:
            x = self._add_intercept(x)

        an = int(self.a * n)
        if an == 0:
            raise ValueError("'a' is too small. Increase 'a' or the number of data points .")

        self.inliers = list(range(n))
        self.predicted_inliers.append(self.inliers)

        for __ in range(self.max_iter):
            #Set up the normal equation
            X_temp=x[self.inliers]
            Y_temp=y[self.inliers]
            B=X_temp.T @ Y_temp
            A=X_temp.T @ X_temp + self.lmbd*self.K 

            #Solve the linear system
            self.coef=sp.linalg.solve(A, B)
            err = np.linalg.norm(y - x @ self.coef, axis=1)

            old_inliers = self.inliers
            self.inliers = np.argpartition(err, an)[:an]
            self.predicted_inliers.append(self.inliers)

            if set(self.inliers) == set(old_inliers):
                break
            
        return self


class Torrent_cv(BaseRobustRegression):
    """Torrent algorithm for regression with robustness to outliers.

    Extends the base regression to implement an iterative process of fitting and refining inliers.

    Torrent_cv performes a cross-validation step after each estimation of the inlinier set the find the best regularization parameter lambda for the next iteration.

    Attributes:
        a (float): Proportion of data considered as inliers.
        max_iter (int): Maximum number of iterations.
        predicted_inliers (list): List to track inliers over iterations.
    """

    def __init__(self, a: float, fit_intercept: bool = False, max_iter: int = 100, K=np.array([0]), lmbd=np.array(0)):
        super().__init__(fit_intercept)
        if not 0 < a < 1:
            raise ValueError("'a' must be in the range (0, 1).")
        self.a = a
        self.max_iter = max_iter
        self.predicted_inliers = []
        self.K=K
        self.lmbd=lmbd


    def fit(self, x: NDArray, y: NDArray) -> Self:
        """Fit model using an iterative process to determine inliers and refit the model.
            lambda: set of regularization parameters over which the cross-valdiation is performed
                    provid only a one-dimensional vector to keep it fixed
            K:      positive semi-definite matrix for the penalty
            """

        n = len(y)
        y = y.reshape(n, -1)

        self._validate_inputs(x, y)
        if self.fit_intercept:
            x = self._add_intercept(x)

        an = int(self.a * n)
        if an == 0:
            raise ValueError("'a' is too small. Increase 'a' or the number of data points .")

        self.inliers = list(range(n))
        self.predicted_inliers.append(self.inliers)
        lambda_cv=self.lmbd[0]
        err_old=np.inf

        for __ in range(self.max_iter):
            X_temp=x[self.inliers]
            Y_temp=y[self.inliers]
            B=X_temp.T @ Y_temp
            A=X_temp.T @ X_temp + lambda_cv*self.K 
            
            coef_new=sp.linalg.solve(A, B)
            err = np.linalg.norm(y - x @ coef_new, axis=1)
            inliers_new = np.argpartition(err, an)[:an]
            err_new=np.linalg.norm(err[inliers_new], ord=2)

            if err_new <= err_old:
                err_old=err_new
                self.coef=coef_new
                self.inliers = inliers_new
                self.predicted_inliers.append(self.inliers)
                lambda_cv=cross_validation(x[self.inliers], y[self.inliers], Lmbd=self.lmbd, K=self.K, a=self.a)    
            else:
               break
   
        return self
    
class Torrent_cv2(BaseRobustRegression):
    """Torrent algorithm for regression with robustness to outliers.

    Extends the base regression to implement an iterative process of fitting and refining inliers.

    Torrent_cv version 2 excecutes the regularized Torrent algroithm and performs at the end a cross-validation to obtain the best regularization parameter lambda.

    Attributes:
        a (float): Proportion of data considered as inliers.
        max_iter (int): Maximum number of iterations.
        predicted_inliers (list): List to track inliers over iterations.
    """
    
    
    def __init__(self, a: float, fit_intercept: bool = False, max_iter: int = 100, K=np.array([0]), lmbd=np.array(0)):
        super().__init__(fit_intercept)
        if not 0 < a < 1:
            raise ValueError("'a' must be in the range (0, 1).")
        self.a = a
        self.max_iter = max_iter
        self.predicted_inliers = []
        self.K=K
        self.lmbd=lmbd

    def fit(self, x: NDArray, y: NDArray) -> Self:
        """Fit model using an iterative process to determine inliers and refit the model.
            lambda: set of regularization parameters over which the cross-valdiation is performed
                    provid only a one-dimensional vector to keep it fixed
            K:      positive semi-definite matrix for the penalty
            """
        k=10        #Number of folds
        n_lambda=len(self.lmbd)
        err_cv=np.zeros(n_lambda)

        #Perform cross-validation
        for i in range(0,n_lambda):
            algo = Torrent_reg(a=self.a, fit_intercept=False, K=self.K, lmbd=self.lmbd[i])
            algo.fit(x,y)
            x_temp=x[algo.inliers]
            y_temp=y[algo.inliers]
            n=len(y_temp)
            fold_size=n//k
            partition=np.random.permutation(n)

            for j in range(0,k):
                test_indx=partition[j*fold_size:(j+1)*fold_size]
                X_train=np.delete(x_temp, test_indx, axis=0)
                Y_train=np.delete(x_temp, test_indx, axis=0)
                B=X_train.T @ Y_train
                A=X_train.T @ X_train + self.lmbd[i]*self.K 
                coef=sp.linalg.solve(A, B)
                err =np.linalg.norm(y[test_indx] - x[test_indx] @ coef, ord=2)**2
                err_cv[i]=err_cv[i]+1/fold_size*err

        #Select the lambda with the smallest cv-error
        lambda_cv=self.lmbd[np.argmin(err_cv)]
        self.lmbd=lambda_cv
        algo = Torrent_reg(a=self.a, fit_intercept=False, K=self.K, lmbd=lambda_cv)
        algo.fit(x, y)

        #Copy the final results to the object
        self.fit_intercept = algo.fit_intercept
        self.inliers = algo.inliers
        self.coef= algo.coef
    
        return self

"""
    simple help function to perform the cross-validation step
"""    

def cross_validation(x, y, Lmbd, K, a) -> float:
    k=10        #Number of folds
    n=len(y)
    fold_size=n//k
    n_lmbd=len(Lmbd)
    partition=np.random.permutation(n)
    err_cv=np.zeros(n_lmbd)
    an_fold = int(a * fold_size)

    for i in range(0, n_lmbd):
        for j in range(0,k):
            test_indx=partition[j*fold_size:(j+1)*fold_size]
            X_train=np.delete(x, test_indx, axis=0)
            Y_train=np.delete(x, test_indx, axis=0)
            B=X_train.T @ Y_train
            A=X_train.T @ X_train + Lmbd[i]*K 
            coef=sp.linalg.solve(A, B)
            err = np.linalg.norm(y[test_indx] - x[test_indx] @ coef, axis=1)
            #inliers_test = np.argpartition(err, an_fold)[:an_fold]
            err=np.linalg.norm(err, ord=2)**2
            err_cv[i]=err_cv[i]+1/fold_size*err

    indx_cv=np.argmin(err_cv)
    return Lmbd[indx_cv]
