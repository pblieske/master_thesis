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
    We test the cross-validation method. For this end, we do the regularization mainly over the parameter lambda 
    and keep the number of coefficients L constant. We compare it to the unregularized torrent with L of order n^(1/2).
    Note that the script can take up to 3 hours to run for m=200, therefore the results are saved.
    To rerun the experiment, set the "run_exp" variable to True.
"""

run_exp=False          # Set to True for running the whole experiment and False to plot an experiment which was already run


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

methods=np.array(["Tor", "Clip", "Omit", "Median", "Clip_1sd", "Omit_1sd", "Median_1sd"])   # Different methods to be teste
n_method=len(methods)                                               # number of methods
colors, ibm_cb = plot_settings()                                    # import colors for plotting
path_results=os.path.join(os.path.dirname(__file__), "results/")    # Path to the results


# ----------------------------------
# run experiments
# ----------------------------------

SEED = 1
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
        err_b=err_boot(transformed=tranformed, a=method_args['a'], lmbd=Lmbd[i], K=K, B=100)
        err_inl[i], err_cap[i], err_m[i]= sum(err_b['err_inl'])/100, sum(err_b['err_cap'])/100, sum(err_b['err_m'])/100
        err_inl_sd[i], err_cap_sd[i], err_m_sd[i]= np.linalg.norm(err_b['err_inl']-err_inl[i])/99, np.linalg.norm(err_b['err_cap']-err_cap[i])/99, np.linalg.norm(err_b['err_m']-err_m[i])/99

    # Get lambda minimizing the estimated error
    ind_cap, ind_inl, ind_m=np.argmin(err_cap), np.argmin(err_inl), np.argmin(err_m)
    lmbd_cap, lmbd_inl, lmbd_m=Lmbd[ind_cap], Lmbd[ind_inl], Lmbd[ind_m]

    # Compute the indices minimizing the estimated perdiction error
    lmbd_cap_1sd, lmbd_inl_1sd, lmbd_m_1sd=Lmbd[min(np.arange(ind_cap,n_lmbd)[err_cap[ind_cap:n_lmbd]>err_cap[ind_cap]+err_cap_sd[ind_cap]])], Lmbd[min(np.arange(ind_inl,n_lmbd)[err_inl[ind_inl:n_lmbd]>err_inl[ind_inl]+err_inl_sd[ind_inl]])], Lmbd[min(np.arange(ind_m,n_lmbd)[err_m[ind_m:n_lmbd]>err_m[ind_m]+err_m_sd[ind_m]])]

    # Run regularized Torrent with the selected regularization parameter
    estimates_tor = get_results(**data_values, a=method_args["a"], method="torrent", basis_type=method_args["basis_type"], L=L)
    estimates_inl = get_results(**data_values, a=method_args["a"], method="torrent_reg", basis_type=method_args["basis_type"], L=L_cv, lmbd=lmbd_inl, K=K)
    estimates_cap = get_results(**data_values, a=method_args["a"], method="torrent_reg", basis_type=method_args["basis_type"], L=L_cv, lmbd=lmbd_cap, K=K)
    estimates_m = get_results(**data_values, a=method_args["a"], method="torrent_reg", basis_type=method_args["basis_type"], L=L_cv, lmbd=lmbd_m, K=K)
    estimates_inl_1sd = get_results(**data_values, a=method_args["a"], method="torrent_reg", basis_type=method_args["basis_type"], L=L_cv, lmbd=lmbd_inl_1sd, K=K)
    estimates_cap_1sd = get_results(**data_values, a=method_args["a"], method="torrent_reg", basis_type=method_args["basis_type"], L=L_cv, lmbd=lmbd_cap_1sd, K=K)
    estimates_m_1sd = get_results(**data_values, a=method_args["a"], method="torrent_reg", basis_type=method_args["basis_type"], L=L_cv, lmbd=lmbd_m_1sd, K=K)

    # Compute the estimation
    y_tor=basis_tor @ estimates_tor["estimate"]
    y_tor=np.ndarray((n_x, 1), buffer=y_tor)
    y_inl, y_cap, y_m= basis_cv @ estimates_inl["estimate"], basis_cv @ estimates_cap["estimate"], basis_cv @ estimates_m["estimate"]
    y_inl_1sd, y_cap_1sd, y_m_1sd= basis_cv @ estimates_inl_1sd["estimate"], basis_cv @ estimates_cap_1sd["estimate"], basis_cv @ estimates_m_1sd["estimate"]
    y_inl, y_cap, y_m=np.ndarray((n_x, 1), buffer=y_inl), np.ndarray((n_x, 1), buffer=y_cap), np.ndarray((n_x, 1), buffer=y_m)
    y_inl_1sd, y_cap_1sd, y_m_1sd=np.ndarray((n_x, 1), buffer=y_inl_1sd), np.ndarray((n_x, 1), buffer=y_cap_1sd), np.ndarray((n_x, 1), buffer=y_m_1sd)

    return [1/n_x*np.linalg.norm(y_true-y_cap, ord=1), 1/n_x*np.linalg.norm(y_true-y_inl, ord=1), 1/n_x*np.linalg.norm(y_true-y_m, ord=1), 1/n_x*np.linalg.norm(y_true-y_tor, ord=1), 1/n_x*np.linalg.norm(y_true-y_cap_1sd, ord=1), 1/n_x*np.linalg.norm(y_true-y_inl_1sd, ord=1), 1/n_x*np.linalg.norm(y_true-y_m_1sd, ord=1)]

# Set up pool
pool=mp.Pool(processes=mp.cpu_count()-1)

# Run Monte Carlo simulation
for i in range(len(noise_vars)):
    print("Noise Variance: ", noise_vars[i])
    res = {"clip": [], "omit": [], "median": [], "tor":[], "clip_1sd":[], "omit_1sd":[], "median_1sd":[]}

    if run_exp:
        for n in num_data:

            print("number of data points: ", n)

            res["clip"].append([])
            res["omit"].append([])
            res["median"].append([])
            res["tor"].append([])
            res["omit_1sd"].append([])
            res["median_1sd"].append([])
            res["clip_1sd"].append([])

            # Get number of coefficients L and construct matirx K for the smothness penalty
            L=max(np.floor(1/L_frac[i]*n**(1/2)).astype(int),2)      
            basis_tor=get_funcbasis(x=test_points, L=L, type=method_args["basis_type"])

            # Run DecoR and DecoR with cross validation for m random sample
            err=pool.starmap(get_err, ((j , n,  y_true, basis_tor, basis_cv, method_args, noise_vars[i], L, L_cv, Lmbd, K) for j in range(m)))

            # Add results to the list
            err = np.array(err).reshape(-1, 7)
            for j in range(m):
                res["clip"][-1].append(err[j, 0])
                res["omit"][-1].append(err[j, 1])
                res["median"][-1].append(err[j, 2])
                res["tor"][-1].append(err[j, 3])
                res["clip_1sd"][-1].append(err[j, 4])
                res["omit_1sd"][-1].append(err[j, 5])
                res["median_1sd"][-1].append(err[j, 6])
           
        # Saving the results to a pickle file
        res["clip"], res["omit"], res["median"], res["tor"], res["omit_1sd"], res["median_1sd"], res["tor_1sd"] = np.array(res["clip"]), np.array(res["omit"]), np.array(res["median"]), np.array(res["tor"]), np.array(res["clip_1sd"]), np.array(res["omit_1sd"]), np.array(res["median_1sd"])
        with open(path_results+"experiment_ridge_decor_final_blocked="+str(noise_vars[i])+'.pkl', 'wb') as fp:
            pickle.dump(res, fp)
            print('Results saved successfully to file.')

    else:
        # Loading the file with the saved results
        with open(path_results+"experiment_ridge_decor_final="+str(noise_vars[i])+'.pkl', 'rb') as fp:
            res = pickle.load(fp)
    
    # Plotting the results
    values = np.concatenate([np.expand_dims(res["tor"], 2),
                            np.expand_dims(res["clip"], 2),
                            np.expand_dims(res["omit"], 2),
                            np.expand_dims(res["median"], 2),
                            np.expand_dims(res["clip_1sd"], 2), 
                            np.expand_dims(res["omit_1sd"], 2),
                            np.expand_dims(res["median_1sd"], 2)],
                             axis=2).ravel()

    time = np.repeat(num_data, m * 7)
    method = np.tile(methods, len(values) // n_method)
    ["Tor", "Clip", "Omit", "Median", "Clip_1sd", "Omit_1sd", "Median_1sd"]

    df = pd.DataFrame({"value": values.astype(float),
                       "n": time.astype(float),
                       "method": method})
    
    err=pd.DataFrame(columns=np.concatenate((['method'], num_data)))

    # Compute the mean error for the not regularized Torrent
    err_add=np.full(len(num_data), np.nan)
    for j in range(len(num_data)):
        err_add[j]=sum((df.loc[(df['n']==num_data[j]) & (df['method']=='Tor')])['value'])/m
    err.loc[0]=['Tor']+list(err_add)

    # Compute the relative improvement
    for l in range(1,n_method):
        err_add=np.full(len(num_data), np.nan)
        for j in range(len(num_data)):
            err_add[j]=(sum((df.loc[(df['n']==num_data[j]) & (df['method']==methods[l])])['value'])/m-err.iat[0, j+1])/err.iat[0, j+1]
        
        err.loc[l]=[methods[l]]+list(err_add)

    print(err)
    df=df.loc[((df['method']=="Tor") |  (df['method']=="Omit")) & (df['n']> 32)]

    sns.lineplot(data=df, x="n", y="value", hue="method", style="method",
                 markers=[ "X", "o"], dashes=False, errorbar=("ci", 95), err_style="band",
                 palette=[colors[i][0], colors[i][1]], legend=True)


pool.close


# ----------------------------------
# plotting
# ----------------------------------

def get_handles():
    point_1 = Line2D([0], [0], label='Torrent', marker='X',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_2 = Line2D([0], [0], label='Regularized', marker='o',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_3 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[0]), markersize=10,
                     color=ibm_cb[1], linestyle='-')
    point_4 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[1]), markersize=10,
                     color=ibm_cb[4], linestyle='-')
    point_5 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[2]), markersize=10,
                     color=ibm_cb[2], linestyle='-')
    return [point_1, point_2, point_3, point_4, point_5]

#Labeling
plt.xlabel("number of data points")
plt.ylabel("$L^1$-error")
plt.title("Regularization Bootstraping")
plt.xscale('log')
plt.xlim(left=num_data[1] - 2)
plt.ylim(-0.1, 2)
plt.hlines(0, num_data[1], num_data[-1], colors='black', linestyles='dashed')
plt.legend(handles=get_handles(), loc="upper right")
plt.tight_layout()

plt.show()