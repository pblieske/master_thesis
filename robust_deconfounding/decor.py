from typing import Optional, Self
from numpy.typing import NDArray
import scipy as sp
import numpy as np

from .robust_regression import BaseRobustRegression


class DecoR:
    """DecoR (Deconfounding with Robust Regression) uses a base robust regression method after transforming the
    data into another basis where confounding is sparse.

    Attributes:
        algo (BaseRobustRegression): The robust regression algorithm to use.
        basis (Optional[NDArray]): Optional basis for transforming the data. If None, Fourier basis is used.
    """

    def __init__(self, algo: BaseRobustRegression, basis: Optional[NDArray] = None) -> None:
        self.basis = basis
        self.algo = algo
        self.xn = None
        self.yn = None

    def _validate_inputs(self, x: NDArray, y: NDArray):
        n = len(y)
        if n != len(x) or n == 0:
            raise ValueError("Data must have the same length and be non-empty.")
        if self.basis is not None and n != len(self.basis):
            raise ValueError("Data and basis must have the same length.")

    def fit(self, x: NDArray, y: NDArray) -> Self:
        """Fit the regression model after transforming the data using a provided basis."""
        self._validate_inputs(x, y)
        n = len(y)

        if self.basis is None:
            self.xn = sp.fft.fft(x.T, norm="forward").T
            self.yn = sp.fft.fft(y, norm="forward")
        else:
            self.xn = self.basis.T @ x / n
            self.yn = self.basis.T @ y / n

        self.algo.fit(self.xn, self.yn)

        return self
    
    def fit_coef(self, x:NDArray, y: NDArray, L:int) -> Self:
        """
            Fit the regression model after transforming the data using the a provided basis.
            Coefficient coresponds to the cosine expansion f(x)=c_0+sum_{k=1}^\infty c_k cos(\pi k x)
        """
        self._validate_inputs(x,y)
        n=len(y)
        if self.basis is None:
            self.xn = sp.fft.fft(x.T, norm="forward").T
            self.yn = sp.fft.fft(y, norm="forward")
        else:
            P_temp = [np.cos(np.pi * x * k) for k in range(L)]
            print(P_temp)
            P = np.hstack(( np.vstack(P_temp))).T
            print(P)
            self.xn = self.basis.T @ P / n
            print(self.x)
            self.yn = self.basis.T @ y / n

        self.algo.fit(self.xn, self.yn)

        return self


    @property
    def estimate(self) -> NDArray:
        """Get the estimated coefficients from the regression model."""
        return self.algo.coef_
