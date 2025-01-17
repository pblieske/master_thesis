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

        """
        #normalize the data
        y_m=np.mean(y)
        y_std=np.std(y)
        y=(y-y_m)/y_std
        d=x.shape[1]
        x_m=np.zeros(d)
        x_std=np.zeros(d)
        for i in range(d):
            x_m[i]=np.mean(x[:,i])
            x_std[i]=np.mean(x[:,i])
            x[:,i]=(x[:,i]-x_m[i])/x_std[i]
        """
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
        
        #Transform the coefficients back
        self.coef=self.model.params  
        """ 
        coef=self.coef
        #coef[0]=y_m+y_std/x_std*x_m*coef[1:(d+1)]
        for i in range(1,d):
            coef[i]=y_std/x_std[i]*coef[i]
        self.coef=coef  
        """
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

        self.coef=self.coef[:,0]    

        return self
    
    
    def cv(self, x: NDArray, y: NDArray, Lmbd: NDArray, k=10) -> dict:
        """
            Estimates the prediction error using a cross-validtion like method.
            Lmbd:   regularization parameters to test
            k:      number of folds
            Returns a dictionary with the estimated prediction error and the set S of stable inliers.
            """
        n=len(y)
        n_lmbd=len(Lmbd)
        err_cv=np.zeros(n_lmbd)

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
            S_i_C=S_i.difference(S)
            n_train=len(S_i_C)
            train_fold_size=n_train//k
            partition_S_C=np.random.permutation(n_train)
            err=0
            for j in range(0,k):
                test_indx=[list(S)[i] for i in partition_S[j*test_fold_size:(j+1)*test_fold_size]]
                train_indx=np.concatenate((np.delete(list(S), partition_S[j*test_fold_size:(j+1)*test_fold_size]), np.delete(list(S_i_C), partition_S_C[j*train_fold_size:(j+1)*train_fold_size])))
                X_train=x[train_indx]
                Y_train=y[train_indx]
                B=X_train.T @ Y_train
                A=X_train.T @ X_train + Lmbd[i]*self.K 
                coef=sp.linalg.solve(A, B)
                err_add = np.linalg.norm(y[test_indx] - x[test_indx] @ coef, ord=2)**2
                err=err+1/n_S*err_add
            err_cv[i]=np.sqrt(err)
    
        return  {"pred_err": err_cv, "S": S}


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
