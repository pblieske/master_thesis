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

        """
        Fit model using an iterative process to determine inliers and refit the model.
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

        self.coef=self.coef[:,0]    

        return self
    
    
    def cv(self, x: NDArray, y: NDArray, Lmbd: NDArray, k=10) -> dict:

        """
            Estimates the prediction error using a cross-validtion like method.
            Lmbd:   regularization parameters to test
            k:      number of folds
            Returns a dictionary with the estimated prediction error "pred_err" and the set "S" of stable inliers across different regularization paramters.
        """
        
        n=len(y)                    #number of data points
        n_lmbd=len(Lmbd)            #number of regularization parameters to test
        err_cv=np.zeros(n_lmbd)     #allocate memory for the estimated prediction error

        #Allocate memory
        estimates=[]   
        S=set(np.arange(0,n))   

        #Run Torrent for all values of lambda to get a stable estimation of an inlinier set
        for i in range(0,n_lmbd):
            algo=Torrent_reg(a=self.a, fit_intercept=False, K=self.K, lmbd=Lmbd[i])
            algo.fit(x,y)
            estimates.append(algo)
            S_i=algo.inliers
            S=S.intersection(S_i)

        #Check that Torrent is stable enought, i.e. S is large enough
        if len(S)==0:
            raise Exception("There is no stable set S.") 
        elif len(S)<n/10:
            print("Warning: S is very small compared to the sample size, |S|=" + str(len(S)))    

        #Perform cross-validation
        n_S=len(S)
        partition_S=np.random.permutation(n_S)
        test_fold_size=n_S//k

        for i in range(0, n_lmbd):
            S_i={inlinier for inlinier in estimates[i].inliers}
            S_i_C=S_i.difference(S)     #indicies in S_i\S
            n_train=len(S_i_C)          #number of observations in S_i\S
            train_fold_size=n_train//k 
            partition_S_C=np.random.permutation(n_train)
            err=0

            for j in range(0,k):
                test_indx=[list(S)[i] for i in partition_S[j*test_fold_size:(j+1)*test_fold_size]]      #index of observations for testing, 
                train_indx=np.concatenate((np.delete(list(S), partition_S[j*test_fold_size:(j+1)*test_fold_size]), np.delete(list(S_i_C), partition_S_C[j*train_fold_size:(j+1)*train_fold_size])))     #index of observations for training
                X_train=x[train_indx]
                Y_train=y[train_indx]
                B=X_train.T @ Y_train
                A=X_train.T @ X_train + Lmbd[i]*self.K 
                coef=sp.linalg.solve(A, B)
                err=err+1/n_S*np.linalg.norm(y[test_indx] - x[test_indx] @ coef, ord=1)
                
            err_cv[i]=err

        return  {"pred_err": err_cv, "S": S}
    
    
    def cv2(self, x: NDArray, y: NDArray, k=10) -> float:

        """
            Estimates the prediction error using cross validation. Since outliers are contained in the test set, we negelect the a largest residuals.
            k: The number of folds
            Returns the estimated generalization error.
        """
        
        n=len(y)            # number of data points
        err=0               # allocate memory for the estimated prediction error
        fold_size=n//k      # cardinality of the folds
        ak = int(self.a * fold_size)

        for j in range(k):
            partition=np.random.permutation(np.arange(n))
            ind=np.array([i for i in partition[j*fold_size:(j+1)*fold_size]]) 
            X_train, Y_train=x[-ind,:], y[-ind].reshape(-1, 1)
            X_test, Y_test=x[ind,:], y[ind].reshape(-1, 1)
            self.fit(X_train, Y_train)
            r = np.linalg.norm(Y_test - X_test @ self.coef, axis=1)
            inliers = np.argpartition(r, ak)[:ak]
            err=err+np.linalg.norm(r[inliers], ord=1)/(k*fold_size)

        return  err