import os, random, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from utils_nonlinear import get_results, get_data, get_conf, plot_settings
from synthetic_data import functions_nonlinear
from robust_deconfounding.utils import get_funcbasis

"""
    We run a short simulation to test how the confidence intervals proposed in the thesis performs, in particualr we want to investigate the coverage.
    The experiment can take several minutes to run, therefore the values are alread saved. To rerun the experiment, set the "run_exp" variable to True.
"""

run_exp=False            # Set to True for running the whole experiment and False to plot an experiment which was already run

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

m = 500                                         # Number of Monte Carlo samples to draw
L=max(np.floor(1/4*n**(1/2)).astype(int),2)     # number of basis functions

colors, ibm_cb = plot_settings()                                    # import colors for plotting
path_results=os.path.join(os.path.dirname(__file__), "results/")    # Path to the results

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
                     

# ----------------------------------
# run experiments or load results
# ----------------------------------

n_x=len(test_points) 
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])
n_alpha=len(alpha)
cov_l, cov_m, cov_h=np.full([n_alpha, n_x], np.nan) , np.full([n_alpha, n_x], np.nan), np.full([n_alpha, n_x], np.nan)

for i in range(n_alpha):
    print('alpha : ' + str(alpha[i]))
    if run_exp:
        T_l,T_m,T_h = np.full([m, n_x], np.nan), np.full([m, n_x], np.nan), np.full([m, n_x], np.nan)
        for _ in tqdm(range(m)):
            # Generate a sample and fit DecoR
            data_values = get_data(n, **data_args, noise_var=noise_vars)
            data_values.pop('u') 
            outlier_points=data_values.pop('outlier_points')
            estimates_decor = get_results(**data_values, **method_args, L=L)
            # Compute the confidence interval
            ci_h=get_conf(x=test_points, **estimates_decor, L=L, alpha=alpha[i], basis_type=method_args["basis_type"], w=0)
            ci_m=get_conf(x=test_points, **estimates_decor, L=L, alpha=alpha[i], basis_type=method_args["basis_type"], w=0.85)
            ci_l=get_conf(x=test_points, **estimates_decor, L=L, alpha=alpha[i], basis_type=method_args["basis_type"], w=1)
            # Check if true f is contained
            T_h[_,:]=(ci_h[:,1]>=y_true[:,0]) & (y_true[:,0]>=ci_h[:,0])
            T_m[_,:]=(ci_m[:,1]>=y_true[:,0]) & (y_true[:,0]>=ci_m[:,0])
            T_l[_,:]=(ci_l[:,1]>=y_true[:,0]) & (y_true[:,0]>=ci_l[:,0])

        #Save the results using a pickle file
        with open(path_results+"confidence_interval_" + '_noise_='+str(noise_vars)+ "_alpha_=" + str(alpha[i]) + '.pkl', 'wb') as fp:
             T={'T_l': T_l, 'T_m': T_m, 'T_h': T_h}
             pickle.dump(T, fp)
    else:
        # Loading the file with the saved results
        with open(path_results+"confidence_interval_" + '_noise_='+str(noise_vars)+ "_alpha_=" + str(alpha[i]) + '.pkl', 'rb') as fp:
            T = pickle.load(fp)
            T_l, T_m, T_h= T['T_l'], T['T_m'], T['T_h']
            m=T_l.shape[0]      # To make sure the number of observations is correct

    # Compute the coverage
    cov_l[i, :], cov_m[i,:], cov_h[i,:]=np.sum(T_l, axis=0)/m, np.sum(T_m, axis=0)/m, np.sum(T_h, axis=0)/m


# ----------------------------------
# Plotting
# ----------------------------------

n_plots=len(test_points)
fig, axs = plt.subplots(1, n_plots ,figsize=(12, 5), layout='constrained')
alpha_min=np.min(alpha)
alpha_max=np.max(alpha)

for i in range(0, n_plots):
    axs[i].plot(alpha, cov_l[ :,i], '--o', color=ibm_cb[1])
    axs[i].plot(alpha, cov_m[ :,i], '--o', color=ibm_cb[2])
    axs[i].plot(alpha, cov_h[ :,i], '--o', color=ibm_cb[4])
    axs[i].plot([alpha_min, alpha_max], [alpha_min, alpha_max], '--', color="black")
    #Labels
    axs[i].set_xlabel('nominal coverage')
    axs[i].set_ylabel('est. actual coverage')
    axs[i].set_title('x=' + str(test_points[i]))

# Legend   
def get_handles():
    point_1 = Line2D([0], [0], label="inliers", marker='o', markersize=10,
                     color=ibm_cb[1], linestyle='--')
    point_2 = Line2D([0], [0], label="weighted", marker='o', markersize=10,
                     color=ibm_cb[2], linestyle='--')
    point_3 = Line2D([0], [0], label="all", marker='o', markersize=10,
                     color=ibm_cb[4], linestyle='--')
    return [point_3, point_2, point_1]

fig.subplots_adjust(right=10)
fig.legend(handles=get_handles(), loc='outside right center')
plt.tight_layout(rect=[0, 0, 0.89, 1.0])
plt.show()