import numpy as np
import random, os, pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from utils_nonlinear import get_results, get_data, plot_settings, err_boot
from synthetic_data import functions_nonlinear
from robust_deconfounding.utils import get_funcbasis

"""
    We test the out-of-bootrstrap generalization error estimations method. For this end, we do the regularization mainly over the parameter lambda 
    and keep the number of coefficients L constant. We compare it to the unregularized torrent with L being of order n^(1/2).
    Note that the script can take several hours to run for m=100, therefore the results are saved in the coresponding folder.
    To rerun the experiment, set the "run_exp" variable to True.
"""

run_exp=False           # Set to True for running the whole experiment and False to plot an experiment which was already run


# ----------------------------------
# Set the parameters
# ----------------------------------

Lmbd_min=10**(-8)       # smallest regularization parameter lambda to be considered
Lmbd_max=10**(1)        # largest regularization paramter lambda to be considered
n_lmbd=100              # number of lambda to test
L_cv=50                 # number of coefficient for the reuglarized torrent
B=100                   # number of sample to draw for the bootstrap
m=100                   # Number of Monte Carlo samples to draw
                                                                           
Lmbd=np.array([np.exp(i/n_lmbd*(np.log(Lmbd_max)-np.log(Lmbd_min))+np.log(Lmbd_min)) for i in range(0, n_lmbd)])      # grid of regularization paramters   
noise_vars = [0, 1, 4]                  # Variance of the noise
num_data = [32, 64, 128, 256, 1024, 8192]   # number of observations n

data_args = {
    "process_type": "uniform",      # "uniform" | "oure"
    "basis_type": "cosine",         # "cosine" | "haar"
    "fraction": 0.25,               # fraction of frequencies that are confounded
    "beta": np.array([2]),      
    "band": list(range(0, 50))      # list(range(0, 50)) | None
}

method_args = {
    "a": 0.7,                       # number of frequencies to remove
    "basis_type": "cosine_cont",    # "cosine_cont" | "cosine_disc" | "poly"
}

methods_plot=np.array(["tor", "omit", "omit_1sd"])              # Three methods that should be plotted
methods=np.array(["tor", "clip", "omit", "median", "clip_1sd", "omit_1sd", "median_1sd"])   # Different methods to be teste
n_method=len(methods)                                               # number of methods
colors, ibm_cb = plot_settings()                                    # import colors for plotting
path_results=os.path.join(os.path.dirname(__file__), "results/")    # Path to the results


# ----------------------------------
# run experiments
# ----------------------------------

SEED = 5
np.random.seed(SEED)
random.seed(SEED)

n_x=200                         # Resolution of x-axis
L_frac=np.array([2, 2, 2])      # Scaling of L for the Torrent
test_points = np.array([i / n_x for i in range(0, n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])

# Compute the basis and regularization matrix K for the smoothness penalty
basis_cv=get_funcbasis(x=test_points, L=L_cv, type=method_args["basis_type"])
diag=np.concatenate((np.array([0]), np.array([i**4 for i in range(1,L_cv+1)])))
K=np.diag(diag)

# Help function needed for the parallelization
def get_err(i, n, y_true, basis_tor, basis_cv, method_args, noise_var, L, L_cv, Lmbd, K):

    # Get the data
    data_values = get_data(n, **data_args, noise_var=noise_var)
    data_values.pop('u', 'basis')
    n_lmbd=len(Lmbd)
    err_inl, err_cap, err_m= np.full([n_lmbd], float(0)), np.full([n_lmbd], float(0)), np.full([n_lmbd], float(0))
    err_inl_sd, err_cap_sd, err_m_sd= np.full([n_lmbd], float(0)), np.full([n_lmbd], float(0)), np.full([n_lmbd], float(0))

    # Basis expansion
    basis=data_values['basis']
    R=get_funcbasis(x=data_values['x'], L=L_cv)
    tranformed={ 'xn': basis.T @ R/ n, 'yn' : basis.T @ data_values['y'] / n}

    # Compute the estimator of DecoR and the regulaized DecoR
    for i in range(n_lmbd):
        err_b=err_boot(transformed=tranformed, a=method_args['a'], lmbd=Lmbd[i], K=K, B=B)
        err_inl[i], err_cap[i], err_m[i]= sum(err_b['err_inl'])/B, sum(err_b['err_cap'])/B, sum(err_b['err_m'])/B
        err_inl_sd[i], err_cap_sd[i], err_m_sd[i]= np.linalg.norm(err_b['err_inl']-err_inl[i])/(B-1), np.linalg.norm(err_b['err_cap']-err_cap[i])/(B-1), np.linalg.norm(err_b['err_m']-err_m[i])/(B-1),

    # Get lambda minimizing the estimated error
    lmbd=np.full([n_method-1], np.nan)
    ind_cap, ind_inl, ind_m=np.argmin(err_cap), np.argmin(err_inl), np.argmin(err_m)
    lmbd[0:3]=np.array([Lmbd[ind_cap], Lmbd[ind_inl], Lmbd[ind_m]])

    # Compute the indices minimizing the estimated perdiction error
    lmbd[3:6]=Lmbd[min(np.arange(ind_cap,n_lmbd)[err_cap[ind_cap:n_lmbd]>err_cap[ind_cap]+err_cap_sd[ind_cap]])], Lmbd[min(np.arange(ind_inl,n_lmbd)[err_inl[ind_inl:n_lmbd]>err_inl[ind_inl]+err_inl_sd[ind_inl]])], Lmbd[min(np.arange(ind_m,n_lmbd)[err_m[ind_m:n_lmbd]>err_m[ind_m]+err_m_sd[ind_m]])]
    e=np.full([n_method], np.nan)       # Allocate memory to save the error

    # Run Torrent and compute the error
    estimates_tor = get_results(**data_values, a=method_args["a"], method="torrent", basis_type=method_args["basis_type"], L=L)
    y_tor=basis_tor @ estimates_tor["estimate"]
    y_tor=np.ndarray((n_x, 1), buffer=y_tor)
    e[0]=1/n_x*np.linalg.norm(y_true-y_tor, ord=1)

    # Compute the error of regularized DecoR for the different regularization parameters lambda
    for k in range(1, n_method):
        estimates=get_results(**data_values, a=method_args["a"], method="torrent_reg", basis_type=method_args["basis_type"], L=L_cv, lmbd=lmbd[k-1], K=K)
        y_est= basis_cv @ estimates["estimate"]
        y_est=np.ndarray((n_x, 1), buffer=y_est)
        e[k]=1/n_x*np.linalg.norm(y_true-y_est, ord=1)

    return e

# Set up pool
pool=mp.Pool(processes=mp.cpu_count()-1)

# Run Monte Carlo simulation
for i in range(len(noise_vars)):
    print("Noise Variance: ", noise_vars[i])
    res = {"clip": [], "omit": [], "median": [], "tor":[], "clip_1sd":[], "omit_1sd":[], "median_1sd":[]}

    if run_exp:
        for n in num_data:
            print("number of data points: ", n)
            for mdth in methods:
                res[mdth].append([])

            # Get number of coefficients L and construct matirx K for the smothness penalty
            L=max(np.floor(1/L_frac[i]*n**(1/2)).astype(int),2)      
            basis_tor=get_funcbasis(x=test_points, L=L, type=method_args["basis_type"])

            # Draw m random samples and compute the error of DecoR and the regularized DecoRs
            err=pool.starmap(get_err, ((j , n,  y_true, basis_tor, basis_cv, method_args, noise_vars[i], L, L_cv, Lmbd, K) for j in range(m)))

            # Add results to the list
            err = np.array(err).reshape(-1, 7)
            for j in range(m):
                for k in range(n_method):
                    res[methods[k]][-1].append(err[j, k])

        # Saving the results to a pickle file
        for mthd in methods:
            res[mthd]=np.array(res[mthd])
    
        with open(path_results+"experiment_ridge_decor_hope="+str(noise_vars[i])+'.pkl', 'wb') as fp:
            pickle.dump(res, fp)
            print('Results saved successfully to file.')

    else:
        # Loading the file with the saved results
        with open(path_results+"experiment_ridge_decor_hope="+str(noise_vars[i])+'.pkl', 'rb') as fp:
            res = pickle.load(fp)
    
    # Compute the relative error and arragne results in a dataframe for plotting
    values=np.array(np.expand_dims(res[methods[0]], 2))
    for mthd in range(1, n_method):
        values=np.concatenate([values, np.expand_dims(res[methods[mthd]], 2)], axis=2)
    values=values.ravel()

    time = np.repeat(num_data, m * n_method)
    method = np.tile(methods, len(values) // n_method)

    df = pd.DataFrame({"value": values.astype(float),
                       "n": time.astype(float),
                       "method": method})
    
    err=pd.DataFrame(columns=np.concatenate((['method'], num_data)))

    # Compute the mean error for the not regularized Torrent
    err_add=np.full(len(num_data), np.nan)
    for j in range(len(num_data)):
        err_add[j]=sum((df.loc[(df['n']==num_data[j]) & (df['method']=='tor')])['value'])/m
    err.loc[0]=['tor']+list(err_add)

    # Compute the relative improvement
    for l in range(1,n_method):
        err_add=np.full(len(num_data), np.nan)
        for j in range(len(num_data)):
            err_add[j]=(sum((df.loc[(df['n']==num_data[j]) & (df['method']==methods[l])])['value'])/m)/err.iat[0, j+1]
        err.loc[l]=[methods[l]]+list(err_add)

    # Print the dataframe contain the results
    print(err)

    # Plotting
    df=df.loc[df['method'].isin(methods_plot) & (df['n']> 32)]
    sns.lineplot(data=df, x="n", y="value", hue="method", style="method",
                 markers=[ "X", "o", "D"], dashes=False, errorbar=("ci", 95), err_style="band",
                 palette=[colors[i][0], colors[i][1]], legend=True)

pool.close


# ----------------------------------
# Labeling the plot
# ----------------------------------

labels={'tor': 'Not regularized', 'clip':'Clipping', 'omit':'Omitting', 'median':'Median', 'clip_1sd':'Clipping 1-S.E.', 'omit_1sd':'Omitting 1-S.E.', 'median_1sd':'Median 1-S.E.' }


def get_handles():
    point_1 = Line2D([0], [0], label=labels[methods_plot[0]], marker='X',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_2 = Line2D([0], [0], label=labels[methods_plot[1]], marker='o',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_3 = Line2D([0], [0], label=labels[methods_plot[2]], marker='D',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_4 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[0]), markersize=10,
                     color=ibm_cb[1], linestyle='-')
    point_5 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[1]), markersize=10,
                     color=ibm_cb[4], linestyle='-')
    point_6 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[2]), markersize=10,
                     color=ibm_cb[2], linestyle='-')
    return [point_1, point_2, point_3, point_4, point_5, point_6]

#Labeling
plt.xlabel("number of data points")
plt.ylabel("$L^1$-error")
plt.title("Regularization with Bootstraping")
plt.xscale('log')
plt.xlim(left=num_data[1] - 2)
plt.ylim(-0.1, 2)
plt.hlines(0, num_data[1], num_data[-1], colors='black', linestyles='dashed')
plt.legend(handles=get_handles(), loc="upper right")
plt.tight_layout()

plt.show()