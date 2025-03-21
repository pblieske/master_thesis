import numpy as np
import scipy as sp
from numpy.typing import NDArray
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import pylab, random


from robust_deconfounding.robust_regression import Torrent, BFS, Torrent_reg
from robust_deconfounding.decor import DecoR
from robust_deconfounding.utils import cosine_basis, haarMatrix, get_funcbasis, get_funcbasis_multivariate
from experiments.synthetic_data import BLPDataGenerator, OUDataGenerator, UniformNonlinearDataGenerator, OUReflectedNonlinearDataGenerator

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


def get_results(x: NDArray, y: NDArray, basis: NDArray, a: float, L: int|NDArray, method: str, basis_type:str, lmbd=0, K=np.array([0]), outlier_points=np.array([])) -> NDArray:
    """
    Estimates the causal coefficient(s) using DecorR with 'method' as robust regression algorithm.

    Args:
        x (NDArray): The data features.
        y (NDArray): The target variable.
        basis (NDArray): The basis for transformation.
        a (float): Hyperparameter for the robust regression method.
        method (str): The method to use for DecoR ("torrent", "bfs", or "ols").
        L (NDArray or int): one-dimensional: number of basis functions to use for the approximation
                            additive model: number of basis functions to use for the approximation for each explenatory variable
    Returns:
        NDArray: The estimated coefficients.

    Raises:
        ValueError: If an invalid method is specified.
        ValueError: If dimensions of x and L dont coincide.
    """

    #Compute the basis
    if isinstance(L, (int, np.int64)):
        R=get_funcbasis(x=x, L=L, type=basis_type)
        n=len(x)
    else:
        R=get_funcbasis_multivariate(x=x, L=L, type=basis_type)
        n=x.shape[0]

    #Running DecoR
    if method[0:3] == "tor":
        if method == "torrent":
            algo = Torrent(a=a, fit_intercept=False)
        elif method == "bfs":
            algo = BFS(a=a, fit_intercept=False)
        elif method == "torrent_reg":
            algo = Torrent_reg(a=a, fit_intercept=False, K=K, lmbd=lmbd)
        else:
            raise ValueError("Invalid method")

        algo = DecoR(algo, basis)
        algo.fit(R, y)

        return {"estimate": algo.estimate, "inliers": algo.inliniers, "transformed": algo.get_transformed}
        
    # Benchamrk methods
    elif method == "ols":
        xn = basis.T @ R / n
        yn = basis.T @ y / n
        model_l = sm.OLS(y, R).fit()
        return {"estimate": model_l.params, "transformed": {"xn":xn, "yn": yn}, "inliers": np.array(range(0, n))}
    elif method == "ridge":
        A=R.T @ R + lmbd * K
        B=R.T @ y
        estimate=sp.linalg.solve(A, B)
        xn = basis.T @ R / n
        yn = basis.T @ y / n
        return{"estimate": estimate, "transformed": {"xn": xn, "yn":yn} }
    elif method == "oracle":
        inliers=np.delete(np.arange(0,n), list(outlier_points))
        xn = basis.T @ R / n
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
    elif process_type=="uniform":
        generator =  UniformNonlinearDataGenerator(basis_type=basis_type, beta=beta, noise_var=noise_var)
    elif process_type=="oure":
        generator= OUReflectedNonlinearDataGenerator(basis_type=basis_type, beta=beta, noise_var=noise_var, noise_type=noise_type)
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
    

def get_conf(x:NDArray, estimate:NDArray, inliers: list, transformed: NDArray, alpha=0.95, L=0, basis_type="cosine_cont", w=0, lmbd=0, K=np.diag(np.array([0]))) -> NDArray:
    """
        Returns a confidence interval for the estimated f evaluated at x.
        Caution: We use all points to estimate the variance (not only the inliers) to avoid a underestimation and 
                to countersteer the fact we only get an interval for \hat{f}
        Arguements:
            x: Points where confidence interval should be evaluated
            estimate: estimated coefficients
            inliers: estimated inliers from DecoR
            alpha: level of coverage for the confidence interval
            w: weight to take into acount the variance, estiamted using only the inliers : (1-w)*variance_large+w*variance_small
        Returns:
            ci=[ci_l, ci_u]: the lower and upper bound for the confidence interval
    """

    xn=transformed["xn"]
    yn=transformed["yn"]

    xn_inl=xn[list(inliers)]
    yn_inl=yn[list(inliers)]
  
    if isinstance(L, (int, np.int64)):
        basis=get_funcbasis(x=x, L=L, type=basis_type)
        L_tot=L
    else:
        basis=get_funcbasis_multivariate(x=x, L=L, type=basis_type)
        L_tot=np.sum(L)+1

    n=xn.shape[0] 
    n_inl=xn_inl.shape[0]

    #Estimate the variance
    r=yn- xn@estimate.T
    df=n-L_tot
    sigma_2=np.sum(np.square(r), axis=0)/df 

    r_inl=yn_inl-xn_inl@estimate.T
    df_inl=n_inl-L_tot
    sigma_2_inl=np.sum(np.square(r_inl), axis=0)/df_inl

    #Compute the linear estimator
    xn=xn[list(inliers)]
    yn=yn[list(inliers)]

    H_help=np.linalg.solve(xn.T @ xn + lmbd*K, xn.T)
    H=basis @ H_help
    sigma=np.sqrt(sigma_2 * np.diag(H @ H.T))
    sigma_inl=np.sqrt(sigma_2_inl * np.diag(H @ H.T))

    #Compute the confidence interval
    qt=sp.stats.t.ppf((1-alpha)/2, df)
    qt_inl=sp.stats.t.ppf((1-alpha)/2, df_inl)

    ci_u=basis@estimate.T - (1-w)*qt*sigma - w * qt_inl*sigma_inl
    ci_l=basis@estimate.T + (1-w)*qt*sigma + w * qt_inl*sigma_inl
    ci=np.stack((ci_l, ci_u), axis=-1)

    return ci


def conf_help(estimate:NDArray, inliers: list, transformed: NDArray, alpha=0.95, lmbd=0, K=np.diag(np.array([0])), L=0)->dict:
    """
        Returns a estimation of the variance sigma and
        Caution: We use all points to estimate the variance (not only the inliers) to avoid a underestimation and 
                to countersteer the fact we only get an interval for \hat{f}
        Arguements:
            x: Points where confidence interval should be evaluated
            estimate: estimated coefficients
            inliers: estimated inliers from DecoR
            alpha: level for the confidence interval
        Returns:
            H: Hat matrix for the coefficients beta
            sigma: estimated variance
            qt: (1-alpha)/2- quantile of the student-t distributions 
    """

    xn=transformed["xn"]
    yn=transformed["yn"]

    if isinstance(L, int):
        n=xn.shape[0]
        L_tot=xn.shape[1]-1
    else:
        n=xn.shape[0]
        L_tot=np.sum(L)+1

    #Estimate the variance
    r=yn- xn@estimate.T
    n=xn.shape[0]
    df=n-L_tot
    xn=transformed["xn"][list(inliers)]

    #Compute results
    qt=sp.stats.t.ppf((1-alpha)/2, df)
    sigma=np.sqrt(np.sum(np.square(r), axis=0)/df)
    H=np.linalg.solve(xn.T @ xn + lmbd*K, xn.T)

    return{'sigma': sigma, 'H':H , 'qt': qt}


def check_eigen(P:NDArray, S: list, G:list, lmbd=0, K=np.array([0])) -> dict:
    """
    Checks if the eigenvalue condition from theorem 4.2 holds.
    Arguments:
        P: tranfromed x variable
        S: estimated set of inliers from decoR
        G: true inliers
        lmbd: regualrization parameter
        K: regualrization matrix
    Returns:
        condition: Boolean indicating if the condition holds
        fraction: fraction on the LHS of the condition which has to be smaller than 1/sqrt(2)
    """

    n=P.shape[0]
    P_S=P[list(S), :]
    G_C=set(np.arange(0,n))-G
    V=S.symmetric_difference(set(G_C))
    P_V=P[list(V), :]

    #Compute eigenvalues
    min=np.min(np.sqrt(np.linalg.eigvals(P_S.T @ P_S + lmbd*K)))
    max=np.max(sp.linalg.svdvals(P_V.T)) if len(V)!=0 else 0 

    #Check the eigenvalue condition and return fraction
    return {'condition': max/min<= 1/np.sqrt(2), 'fraction': max/min}



def bootstrap(x_test:NDArray, transformed:NDArray,  a: float, L: int|NDArray, basis_type:str, M=500) -> NDArray:
    """
    Checks if the eigenvalue condition from theorem 4.2 holds.
    Arguments:
        x_test: points to evaluate
        transformed: transformed sample
        a: paramter for Torrent
        L: number of basis functions for the approximation of f
        basis_type: type of basis for the approximation of f
        M: number of bootstrap samples to draw
    Returns:
        boot: bootstraps, evaluated at x_test
    """

    xn=transformed["xn"]
    yn=transformed["yn"]
    n=xn.shape[0]
    basis=get_funcbasis(x=x_test, L=L, type=basis_type)
    boot= np.full([M, len(x_test)], np.nan)

    for _ in range(M):
        ind=np.random.choice(np.arange(n), size=n, replace=True)
        R, y=xn[ind,:], yn[ind]
        algo = Torrent_reg(a=a, fit_intercept=False, lmbd=0, K=np.array([0]))
        algo.fit(R, y)
        boot[_, :]=basis @ algo.coef
    
    boot=np.sort(boot, axis=0)

    return boot


def double_bootstrap(x_test:NDArray, transformed:NDArray, estimate:NDArray, a: float, L: int|NDArray, basis_type:str, M=100, B=200) -> dict:
    """
    Returns an esimtation of the coverage using double bootstraping.
    Arguments:
        x_test: points to evaluate
        transformed: transformed sample
        a: paramter for Torrent
        L: number of basis functions for the approximation of f
        basis_type: type of basis for the approximation of f
        M: number of outer bootstrap samples to draw
        B: number of inner bootstrap samples to draw
    Returns: dictonary {'nominal': nominal_alpha, 'actual': actual_alpha}
        nominal_alpha: original, nominal coverage
        acutal_alpha: the estimated acutal coverage
    """


    xn=transformed["xn"]
    yn=transformed["yn"]
    M=int(2*np.ceil(M/2))    #Make sure that M is even to keep it simple
    n=xn.shape[0]
    n_alpha=int(B/2)

    # Compute initail estimation
    basis=get_funcbasis(x=x_test, L=L, type=basis_type)
    y_est=basis @ estimate

    # Allocate memory
    nominal_alpha=np.array([2*i / B for i in range(int(B/2))])
    cov=np.full([M, int(B/2), len(x_test)], np.nan)

    for i in range(M):
        # Draw a first level bootstrap sample
        ind=np.random.choice(np.arange(n), size=n, replace=True)
        R, y=xn[ind,:], yn[ind]
        algo1 = Torrent_reg(a=a, fit_intercept=False, lmbd=0, K=np.array([0]))
        algo1.fit(R, y)
        y_1=basis@ algo1.coef

        # Draw second level bootstraps
        boot_double=np.full([B, len(x_test)], np.nan)
        for j in range(B):
            ind_double=np.random.choice(np.arange(n), size=n, replace=True)
            R_double, y_double = R[ind_double,:], y[ind_double]
            algo = Torrent_reg(a=a, fit_intercept=False, lmbd=0, K=np.array([0]))
            algo.fit(R_double, y_double)
            boot_double[j, :]=basis @ algo.coef

        # Compute confidence interval of the second bootstrap level
        boot_double=np.sort(boot_double, axis=0)
        cov[i,:, :]=(2*y_1-boot_double[range(B-1,int(B/2)-1, -1),:]<= np.repeat([y_est],  int(B/2), axis=0)) &  (np.repeat([y_est],  int(B/2), axis=0) <= 2*y_1-boot_double[0:int(B/2), :])

    # Compute the estimated, actual coverage
    actual_alpha=np.full([n_alpha, len(x_test)], np.nan)
    for i in range(n_alpha):
        actual_alpha[i, :]=np.sum(cov[:,(n_alpha-1-i),:], axis=0)/M

    return {'nominal': nominal_alpha, 'actual': actual_alpha}


def err_boot( transformed:NDArray, a: float, lmbd=0, K=np.array([0]), B=100) -> dict:
    """
    Estimate the prediction error on the transformed smaple using a out-of bootstrap estimation.
    Arguemnts:
        transformed: transformed sample
        a: number of outliers to remove
        lmbd: regularization parameter
        K: regualrization matrix
        B: number of bootstrap samples to draw
    Returns:
        err_inl: out-of bootstrap estimation using the a smallest residuals
        err_cap: out-of bootstrap estimation clipping the residuals at the c-a largest value
        err_m: out of bootstrap estimation using the median residual
    """

    xn=transformed["xn"]
    yn=transformed["yn"]
    n=xn.shape[0]
    err_inl, err_cap, err_m=np.full([B], np.nan), np.full([B], np.nan), np.full([B], np.nan)

    for _ in range(B):
        # Draw sample with replacement
        ind=np.random.choice(np.arange(n), size=n, replace=True)
        R_train, y_train=xn[ind,:], yn[ind]
        R_test, y_test=xn[-ind, :], yn[-ind].reshape(-1, 1)
        l=len(y_test)
        al=int(l*a)

        # Fit regularized Torrent
        algo = Torrent_reg(a=a, fit_intercept=False, lmbd=lmbd, K=K)
        algo.fit(R_train, y_train)
        r = np.linalg.norm(y_test - (R_test @ algo.coef).reshape(-1,1), axis=1)

        # Compute error using hard thersholding
        m=np.sort(r)[al]
        r_cap=np.minimum(r, np.full([l], m))
        err_cap[_]=np.linalg.norm(r_cap, ord=1)/(B*l)

        # Compute error neglecating the largests residuals
        inliers = np.argpartition(r, al)[:al]
        err_inl[_]= np.linalg.norm(r[inliers], ord=1)/(B*al)

        # Compute the median error
        err_m[_]=np.sort(r)[int(l/2)]/B

    return {'err_inl': err_inl, 'err_cap': err_cap, 'err_m': err_m}