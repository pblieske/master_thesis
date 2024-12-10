import numpy as np
import scipy as sp
from numpy.typing import NDArray
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import pylab


import sys
sys.path.insert(0, '/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/code/master_thesis')

from robust_deconfounding.robust_regression import Torrent, BFS, Torrent_reg, Torrent_cv, Torrent_cv2, Torrent_cv3
from robust_deconfounding.decor import DecoR
from robust_deconfounding.utils import cosine_basis, haarMatrix
from experiments.synthetic_data import BLPDataGenerator, OUDataGenerator, BLPNonlinearDataGenerator

def plot_settings():
    """
    Sets plot configuration parameters for a consistent look across plots.

    Returns:
        tuple[list[list[str]], list[str]]: A tuple containing color palettes and a list of colors.

    Reference: https://lospec.com/palette-list/ibm-color-blind-safe
    """
    size = 12
    params = {
        'legend.fontsize': size,
        'legend.title_fontsize': size,
        'figure.figsize': (5, 5),
        'axes.labelsize': size,
        'axes.titlesize': size,
        'xtick.labelsize': size,
        'ytick.labelsize': size
    }
    pylab.rcParams.update(params)

    ibm_cb = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000", "#808080"]
    return [[ibm_cb[1], ibm_cb[1]], [ibm_cb[4], ibm_cb[4]], [ibm_cb[2], ibm_cb[2]]], ibm_cb


def r_squared(x: NDArray, y_true: NDArray, beta: NDArray) -> float:
    y_pred = x @ beta
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return 1-u/v


def get_results(x: NDArray, y: NDArray, basis: NDArray, a: float, L: int, method: str, lmbd=0, K=np.array([0])) -> NDArray:
    """
    Estimates the causal coefficient(s) using DecorR with 'method' as robust regression algorithm.

    Args:
        x (NDArray): The data features.
        y (NDArray): The target variable.
        basis (NDArray): The basis for transformation.
        a (float): Hyperparameter for the robust regression method.
        method (str): The method to use for DecoR ("torrent", "bfs", or "ols").

    Returns:
        NDArray: The estimated coefficients.

    Raises:
        ValueError: If an invalid method is specified.
    """
    if method[0:3] == "tor":
        if method == "torrent":
            algo = Torrent(a=a, fit_intercept=False)
        elif method == "bfs":
            algo = BFS(a=a, fit_intercept=False)
        elif method == "torrent_reg":
            algo = Torrent_reg(a=a, fit_intercept=False, K=K, lmbd=lmbd)
        elif method =="torrent_cv":
            robust_algo = Torrent_reg(a=a, fit_intercept=False, K=K, lmbd=0)
            algo = DecoR(algo=robust_algo, basis=basis)
            algo.fit_coef(x=x, y=y, L=L)
            trans=algo.get_transformed
            P_n=trans["xn"]
            y_n=trans["yn"]
            #Get CV values
            k=10    #Number of folds
            cv=robust_algo.cv(x=P_n, y=y_n, Lmbd=lmbd, k=k)
            err_cv=cv["pred_err"]
            S=cv["S"]
            lmbd_cv=lmbd[np.argmin(err_cv)]
            #Set the algorithm to Torrent with the selected lmbd_cv
            algo = Torrent_reg(a=a, fit_intercept=False, K=K, lmbd=lmbd_cv)
        else:
            raise ValueError("Invalid method")
        """
        elif method =="torrent_cv2":
            algo = Torrent_cv2(a=a, fit_intercept=False, K=K, lmbd=lmbd)
        elif method =="torrent_cv3":
            algo = Torrent_cv3(a=a, fit_intercept=False, K=K, lmbd=lmbd)
        """

        algo = DecoR(algo, basis)
        algo.fit_coef(x, y, L)

        if method =="torrent_cv":
            return {"estimate": algo.estimate, "inliniers": algo.inliniers, "transformed": algo.get_transformed, "S":S}
        else:
            return {"estimate": algo.estimate, "inliniers": algo.inliniers, "transformed": algo.get_transformed}

    elif method == "ols":
        n=len(x)
        P_temp = [np.cos(np.pi * x.T * k) for k in range(L)]
        P =  np.vstack(P_temp).T
        xn = basis.T @ P / n
        yn = basis.T @ y / n
        model_l = sm.OLS(yn, xn).fit()
        return model_l.params
    elif method == "ridge":
        n=len(x)
        P_temp = [np.cos(np.pi * x.T * k) for k in range(L)]
        P =  np.vstack(P_temp).T
        xn = basis.T @ P / n
        yn = basis.T @ y / n
        A=xn.T @ xn + lmbd * K
        B=xn.T @ yn
        estimate=sp.linalg.solve(A, B)
        return{"estimate": estimate, "transformed": {"xn": xn, "yn":yn} }
    else:
        raise ValueError("Invalid method")


def get_data(n: int, process_type: str, basis_type: str, fraction: float, beta: NDArray, noise_var: float,
             band: list) -> dict:
    """
    Generates data for deconfounding experiments with different settings.

    Args:
        n (int): Number of data points.
        process_type (str): Type of data generation process ("ou", "blp", "blpnl").
        basis_type (str): Type of basis transformation ("cosine" or "haar").
        fraction (float): Fraction of outliers in the data.
        beta (NDArray): True coefficient vector for the linear relationship.
        noise_var (float): Variance of the noise added to the data.
        band (list): Frequency band for concentrated confounding (BLP process only).

    Returns:
        dict: A dictionary containing generated data (x, y), and the basis matrix.

    Raises:
        ValueError: If an invalid process type or basis type is specified.
    """
    if process_type == "ou":
        generator = OUDataGenerator(basis_type=basis_type, beta=beta, noise_var=noise_var)
    elif process_type == "blp":
        generator = BLPDataGenerator(basis_type=basis_type, beta=beta, noise_var=noise_var, band=band)
    elif process_type=="blpnl":
        generator = BLPNonlinearDataGenerator(basis_type=basis_type, beta=beta, noise_var=noise_var, band=band, fraction=fraction)
    else:
        raise ValueError("process_type not implemented")

    if basis_type == "cosine":
        basis = cosine_basis(n)
    elif basis_type == "haar":
        basis = haarMatrix(n)
    else:
        raise ValueError("basis not implemented")

    n_outliers = int(fraction*n)
    outlier_points = np.array([1]*n_outliers + [0]*(n - n_outliers)).reshape(-1, 1)
    np.random.shuffle(outlier_points)

    if beta.shape[0] == 2:
        x, y = generator.generate_data_2_dim(n=n, outlier_points=outlier_points)
    else:
        x, y, u = generator.generate_data(n=n, outlier_points=outlier_points)

    return {"x": x, "y": y, "u": u, "basis": basis}


def plot_results(res: dict, num_data: list, m: int, colors) -> None:
    """
    Plots the estimated coefficients using DecoR and OLS methods across different data sizes.

    Args:
        res (dict): A dictionary containing estimated coefficients for DecoR and OLS.
        num_data (list): A list of data sizes used in the experiments.
        m (int): Number of repetitions for each data size.
        colors (list): A list of colors for plotting the methods.
    """
    values = np.concatenate([np.expand_dims(res["ols"], 2),
                             np.expand_dims(res["DecoR"], 2)], axis=2).ravel()

    time = np.repeat(num_data, m * 2)
    method = np.tile(["OLS", "DecoR"], len(values) // 2)

    df = pd.DataFrame({"value": values.astype(float),
                       "n": time.astype(float),
                       "method": method})

    sns.lineplot(data=df, x="n", y="value", hue="method", style="method",
                 markers=["o", "X"], dashes=False, errorbar=("ci", 95), err_style="band",
                 palette=[colors[0], colors[1]], legend=True)

def estimated_pred_err(x, y, lmbd=0, K=np.array([0]), k=10) -> float:
     
    """
        Performs a k-fold cross-validation to estimate the prediction error.
    """
    n=len(y)
    fold_size=n//k
    partition=np.random.permutation(n)
    err=0

    for j in range(0,k):
        test_indx=partition[j*fold_size:(j+1)*fold_size]
        X_train=np.delete(x, test_indx, axis=0)
        Y_train=np.delete(x, test_indx, axis=0)
        B=X_train.T @ Y_train
        A=X_train.T @ X_train + lmbd*K 
        coef=sp.linalg.solve(A, B)
        err_add=np.linalg.norm(y[test_indx] - x[test_indx] @ coef, ord=2)**2
        err=err+1/n*err_add

    return np.sqrt(err)
