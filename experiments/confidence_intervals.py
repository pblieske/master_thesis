import os, random, pickle
import numpy as np
import matplotlib.pyplot as plt
from utils_nonlinear import get_results, get_data, get_conf, plot_settings
from synthetic_data import functions_nonlinear
from robust_deconfounding.utils import get_funcbasis

"""
    We run a short simulation to test how the confidence intervals proposed in the thesis performs, in particualr we want to investigate the coverage.
"""

run_exp=True            # Set to True for running the whole experiment and False to plot an experiment which was already run

# ----------------------------------
# Parameters
# ----------------------------------

test_points=np.array([0.1, 0.5, 0.9])                           # points for which the results are plotted
alpha=np.array([0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]) # covergae to comute the confidence intervals 
noise_vars = 1          # Variance of the noise
n = 2**10               # number of observations n

data_args = {
    "process_type": "uniform",      # "uniform" | "oure"
    "basis_type": "cosine",         # "cosine" | "haar"
    "fraction": 0.25,               # fraction of frequencies that are confounded
    "beta": np.array([3]),      
    "band": list(range(0, 50))      # list(range(0, 50)) | None
}

method_args = {
    "a": 0.7,                       # number of frequencies to remove
    "method": "torrent",            # "torrent" | "torrent_reg"
    "basis_type": "cosine_cont",    # "cosine_cont" | "cosine_disc" | "poly"
}

m = 200                                         # Number of Monte Carlo samples to draw
L=max(np.floor(1/4*n**(1/2)).astype(int),2)     # number of basis functions

colors, ibm_cb = plot_settings()    # import colors for plotting
path_results=os.path.join(os.path.dirname(__file__), "results/")    # Path to the results

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
                     

# ----------------------------------
# run experiments or loading results
# ----------------------------------

n_x=len(test_points)             #Resolution of x-axis
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])
n_alpha=len(alpha)
cov=np.full([n_alpha, n_x], np.nan)

for i in range(n_alpha):
    print('alpha : ' + str(alpha[i]))
    if run_exp:
        T = np.full([m, n_x], np.nan)
        for _ in range(m):
            data_values = get_data(n, **data_args, noise_var=noise_vars)
            data_values.pop('u') 
            outlier_points=data_values.pop('outlier_points')
            estimates_decor = get_results(**data_values, **method_args, L=L)
            ci=get_conf(x=test_points, **estimates_decor, L=L, alpha=alpha[i], basis_type=method_args["basis_type"], small=True)
            T[_,:]=(ci[:,1]>=y_true[:,0]) & (y_true[:,0]>=ci[:,0])

        #Save the results using a pickle file
        with open(path_results+"confidence_interval_" + '_noise_='+str(noise_vars)+ "_alpha_=" + str(alpha[i]) + '.pkl', 'wb') as fp:
            pickle.dump(T, fp)
    else:
        # Loading the file with the saved results
        with open(path_results+"confidence_interval_" + '_noise_='+str(noise_vars)+ "_alpha_=" + str(alpha[i]) + '.pkl', 'rb') as fp:
            T = pickle.load(fp)
    # Compute the coverage
    cov[i, :]=np.sum(T, axis=0)/m


# ----------------------------------
# Plotting
# ----------------------------------

n_plots=len(test_points)
fig, axs = plt.subplots(1, n_plots)
fig.set_size_inches(10,5)
alpha_min=np.min(alpha)
alpha_max=np.max(alpha)

for i in range(0, n_plots):
    axs[i].plot(alpha, cov[ :,i], '--o', color="black")
    axs[i].plot([alpha_min, alpha_max], [alpha_min, alpha_max], '--', color=ibm_cb[2])
    #Labels
    axs[i].set_xlabel('nominal coverage')
    axs[i].set_ylabel('est. actual coverage')
    axs[i].set_title('x=' + str(test_points[i]))
    
plt.tight_layout()
plt.show()