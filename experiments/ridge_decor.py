import numpy as np
import random, os, pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.lines import Line2D

from utils_nonlinear import get_results, get_data, plot_settings, plot_results, err_boot
from synthetic_data import functions_nonlinear
from robust_deconfounding.utils import get_funcbasis

"""
    We test the cross-validation method. For this end, we do the regularization mainly over the parameter lambda 
    and keep the number of coefficients L constant. We compare it to the unregularized torrent with L of order n^(1/2).
    Note that the script can take up to 3 hours to run for m=200, therefore the results are saved.
    To rerun the experiment, set the "run_exp" variable to True.
"""

run_exp=False         # Set to True for running the whole experiment and False to plot an experiment which was already run


# ----------------------------------
# Set the parameters
# ----------------------------------

Lmbd_min=10**(-8)       # smallest regularization parameter lambda to be considered
Lmbd_max=10**(1)        # largest regularization paramter lambda to be considered
n_lmbd=100              # number of lambda to test
L_cv=30                 # number of coefficient for the reuglarized torrent
m=100                   # Number of Monte Carlo samples to draw
                                                                           
Lmbd=np.array([np.exp(i/n_lmbd*(np.log(Lmbd_max)-np.log(Lmbd_min))+np.log(Lmbd_min)) for i in range(0, n_lmbd)])      # grid of regularization paramters   
noise_vars = [0, 1, 4]                       # Variance of the noise
num_data = [ 64, 128, 256, 1024, 8192]       # number of observations n

data_args = {
    "process_type": "uniform",      # "uniform" | "oure"
    "basis_type": "cosine",         # "cosine" | "haar"
    "fraction": 0.25,               # fraction of frequencies that are confounded
    "beta": np.array([2]),      
    "band": list(range(0, 50))      # list(range(0, 50)) | None
}

method_args = {
    "a": 0.7,                       # number of frequencies to remove
    "method": "torrent_reg",        # "torrent" | "torrent_reg"
    "basis_type": "cosine_cont",    # "cosine_cont" | "cosine_disc" | "poly"
}

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
    err_inl, err_cap= np.full([n_lmbd], float(0)), np.full([n_lmbd], float(0))

    # Basis expansion
    basis=data_values['basis']
    R=get_funcbasis(x=data_values['x'], L=L_cv)
    tranformed={ 'xn': basis.T @ R/ n, 'yn' : basis.T @ data_values['y'] / n}

    # Compute the estimator of DecoR and the regulaized DecoR
    for i in range(n_lmbd):
        err_b=err_boot(transformed=tranformed, a=method_args['a'], lmbd=Lmbd[i], K=K, B=100)
        err_inl[i], err_cap[i]= err_b['err_inl'], err_b['err_cap']

    # Get lambda minimizing the estimated error
    lmbd_cap=Lmbd[np.argmin(err_cap)]
    lmbd_inl=Lmbd[np.argmin(err_inl)]

    # Run regularized Torrent with the selected regularization parameter
    estimates_tor = get_results(**data_values, a=method_args["a"], method="torrent", basis_type=method_args["basis_type"], L=L)
    estimates_binl = get_results(**data_values, a=method_args["a"], method="torrent_reg", basis_type=method_args["basis_type"], L=L_cv, lmbd=lmbd_inl, K=K)
    estimates_bcap = get_results(**data_values, a=method_args["a"], method="torrent_reg", basis_type=method_args["basis_type"], L=L_cv, lmbd=lmbd_cap, K=K)
   
    # Compute the estimation
    y_tor=basis_tor @ estimates_tor["estimate"]
    y_tor=np.ndarray((n_x, 1), buffer=y_tor)
    y_binl, y_bcap= basis_cv @ estimates_binl["estimate"], basis_cv @ estimates_bcap["estimate"]
    y_binl, y_bcap=np.ndarray((n_x, 1), buffer=y_binl), np.ndarray((n_x, 1), buffer=y_bcap)

    # (bcap is at the place of DecoR)
    return [1/n_x*np.linalg.norm(y_true-y_bcap, ord=1), 1/n_x*np.linalg.norm(y_true-y_binl, ord=1)]

# Set up pool
pool=mp.Pool(processes=mp.cpu_count()-1)

# Run Monte Carlo simulation
for i in range(len(noise_vars)):
    print("Noise Variance: ", noise_vars[i])
    res = {"DecoR": [], "ols": []}

    if run_exp:
        for n in num_data:
            res["DecoR"].append([])
            res["ols"].append([])

            # Get number of coefficients L and construct matirx K for the smothness penalty
            L=max(np.floor(1/L_frac[i]*n**(1/2)).astype(int),2)      
            basis_tor=get_funcbasis(x=test_points, L=L, type=method_args["basis_type"])
            print("number of data points: ", n)
            print("number of coefficients: ", L)

            # Run DecoR and DecoR with cross validation for m random sample
            err=pool.starmap(get_err, ((j , n,  y_true, basis_tor, basis_cv, method_args, noise_vars[i], L, L_cv, Lmbd, K) for j in range(m)))

            # Add results to the list
            err = np.array(err).reshape(-1, 2)
            for j in range(m):
                res["ols"][-1].append(err[j, 0])
                res["DecoR"][-1].append(err[j, 1])
           
        # Saving the results to a pickle file
        res["DecoR"], res["ols"] = np.array(res["DecoR"]), np.array(res["ols"])
        with open(path_results+"experiment_err_boot_="+str(noise_vars[i])+'.pkl', 'wb') as fp:
            pickle.dump(res, fp)
            print('Results saved successfully to file.')

    else:
        # Loading the file with the saved results
        with open(path_results+"experiment_err_boot_="+str(noise_vars[i])+'.pkl', 'rb') as fp:
            res = pickle.load(fp)
    
    # Plotting the results
    plot_results(res, num_data, m, colors=colors[i])

pool.close

# ----------------------------------
# plotting
# ----------------------------------

def get_handles():
    point_1 = Line2D([0], [0], label='cap', marker='o',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_2 = Line2D([0], [0], label='inl', marker='X',
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
plt.xlim(left=num_data[0] - 2)
plt.ylim(-0.1, 2.5)
plt.hlines(0, num_data[0], num_data[-1], colors='black', linestyles='dashed')
plt.legend(handles=get_handles(), loc="upper right")
plt.tight_layout()

plt.show()