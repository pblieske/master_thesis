import numpy as np
import scipy as sp
from numpy.typing import NDArray
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import pylab
import matplotlib.pyplot as plt


import sys
sys.path.insert(0, '/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/code/master_thesis')

from robust_deconfounding.robust_regression import Torrent, BFS, Torrent_reg
from robust_deconfounding.decor import DecoR
from robust_deconfounding.utils import cosine_basis, haarMatrix, get_funcbasis
from experiments.synthetic_data import BLPDataGenerator, OUDataGenerator, BLPNonlinearDataGenerator, OUNonlinearDataGenerator

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


def get_results(x: NDArray, y: NDArray, basis: NDArray, a: float, L: int, method: str, basis_type:str, lmbd=0, K=np.array([0]), outlier_points=np.array([]), normalize=True) -> NDArray:
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
        x=get_funcbasis(x=x, L=L, type=basis_type)
        if method == "torrent":
            algo = Torrent(a=a, fit_intercept=False)
        elif method == "bfs":
            algo = BFS(a=a, fit_intercept=False)
        elif method == "torrent_reg":
            algo = Torrent_reg(a=a, fit_intercept=False, K=K, lmbd=lmbd)
        elif method =="torrent_cv":
            robust_algo = Torrent_reg(a=a, fit_intercept=False, K=K, lmbd=0)
            algo = DecoR(algo=robust_algo, basis=basis)
            algo.fit(x=x, y=y)
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

        algo = DecoR(algo, basis)
        algo.fit(x, y)

        if method =="torrent_cv":
            return {"estimate": algo.estimate, "inliers": algo.inliniers, "transformed": algo.get_transformed, "S":S}
        else:
            return {"estimate": algo.estimate, "inliers": algo.inliniers, "transformed": algo.get_transformed}

    elif method == "ols":
        n=len(x)
        P=get_funcbasis(x=x, L=L, type=basis_type)
        xn = basis.T @ P / n
        yn = basis.T @ y / n
        model_l = sm.OLS(y, P).fit()
        return {"estimate": model_l.params, "transformed": {"xn":xn, "yn": yn}, "inliers": np.array(range(0, n))}
    elif method == "ridge":
        n=len(x)
        P=get_funcbasis(x=x, L=L, type=basis_type)
        A=P.T @ P + lmbd * K
        B=P.T @ y
        estimate=sp.linalg.solve(A, B)
        xn = basis.T @ P / n
        yn = basis.T @ y / n
        return{"estimate": estimate, "transformed": {"xn": xn, "yn":yn} }
    elif method == "oracle":
        n=len(x)
        inliers=np.delete(np.arange(0,n), list(outlier_points))
        P=get_funcbasis(x=x, L=L, type=basis_type)
        xn = basis.T @ P / n
        yn = basis.T @ y / n
        model_l = sm.OLS(yn[inliers], xn[inliers]).fit()
        return {"estimate": model_l.params, "transformed": {"xn":xn, "yn": yn}, "inliers":inliers}
    else:
        raise ValueError("Invalid method")


def get_data(n: int, process_type: str, basis_type: str, fraction: float, beta: NDArray, noise_var: float,
             band: list, noise_type="normal") -> dict:
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
        generator = BLPNonlinearDataGenerator(basis_type=basis_type, beta=beta, noise_var=noise_var, band=band, noise_type=noise_type)
    elif process_type=="ounl":
        generator = OUNonlinearDataGenerator(basis_type=basis_type, beta=beta, noise_var=noise_var, noise_type=noise_type)
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

    outliers={outlier_points[i,0]*i for i in np.arange(0,n)}
    if outlier_points[0]==0:
        outliers=outliers-{0}

    return {"x": x, "y": y, "u": u, "basis": basis, "outlier_points": outliers}


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
    
def plot_results_2yaxis(res: dict, num_data: list, m: int, colors, first=False) -> None:
    """
    Plots the estimated coefficients using DecoR and OLS methods across different data sizes.

    Args:
        res (dict): A dictionary containing estimated coefficients for DecoR and OLS.
        num_data (list): A list of data sizes used in the experiments.
        m (int): Number of repetitions for each data size.
        colors (list): A list of colors for plotting the methods.
    """
    values = np.concatenate([np.expand_dims(res["ols"], 2)], axis=2).ravel()

    time = np.repeat(num_data, m)
    method = np.tile(["OLS"], len(values))

    df = pd.DataFrame({"value": values.astype(float),
                       "n": time.astype(float),
                       "method": method})

    sns.lineplot(data=df, x="n", y="value", hue="method", style="method",
                 markers=["o"], dashes=False, errorbar=("ci", 95), err_style="band",
                 palette=[colors[0]], legend=True)
    

    values = np.concatenate([np.expand_dims(res["DecoR"], 2)], axis=2).ravel()
    method = np.tile(["DecoR"], len(values))

    df = pd.DataFrame({"value": values.astype(float),
                       "n": time.astype(float),
                       "method": method})
    if first:
        ax2 = plt.twinx()
    sns.lineplot(data=df, x="n", y="value", hue="method", style="method",
                 markers=["X"], dashes=False, errorbar=("ci", 95), err_style="band",
                 palette=[colors[1]], legend=True, ax=ax2)       


def get_conf(x:NDArray, estimate:NDArray, inliers: list, transformed: NDArray, alpha=0.95, lmbd=0, K=np.diag(np.array([0])), basis_type="cosine_cont") -> NDArray:
    """
        Returns a confidence interval for the estimated f evaluated at x, assuming that S contains only inliers.
        Problem: In our sample there is a bias present introduced by cutting the series of at L.
    """

    xn=transformed["xn"]#[list(inliers)]
    yn=transformed["yn"]#[list(inliers)]

    #Estimate the variance
    r=yn- xn@estimate.T
    n=xn.shape[0]
    L=xn.shape[1]
    df=n-L
    sigma_2=np.sum(np.square(r), axis=0)/df 

    #Compute the linear estimator
    xn=xn[list(inliers)]
    yn=yn[list(inliers)]
    basis=get_funcbasis(x=x, L=L, type=basis_type)
    H=basis @ np.linalg.inv(xn.T @ xn + lmbd*K) @ xn.T
    sigma=np.sqrt(sigma_2*np.diag(H @ H.T))

    #Compute the confidence interval
    qt=sp.stats.t.ppf((1-alpha)/2, df)
    ci_u=basis@estimate.T - qt*sigma 
    ci_l=basis@estimate.T + qt*sigma 
    ci=np.stack((ci_l, ci_u), axis=-1)

    return ci


def check_eigen(x:NDArray, S: list, G:list, lmbd=0, K=np.array([0])) -> dict:
    """
    Checks if the eigenvalue condition from theorem 4.2 holds.
    S: estimated set of inliers from decoR
    G: true inliers
    lmbd: regualrization parameter
    K: regualrization matrix
    """
    n=x.shape[0]
    x_S=x[list(S), :]
    G_C=set(np.arange(0,n))-G
    V=S.symmetric_difference(set(G_C))
    x_V=x[list(V), :]
    #Compute eigenvalues
    min=np.min(np.sqrt(np.linalg.eigvals(x_S.T @ x_S + lmbd*K)))
    max=np.max(sp.linalg.svdvals(x_V.T)) if len(V)!=0 else 0 #np.max(np.sqrt(np.linalg.eigvals(x_V.T @ x_V))) if len(V)!=0 else 0 
    #Check the eigenvalue condition
    return {'condition': max/min<= 1/np.sqrt(2), 'fraction': max/min}