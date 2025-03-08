import os, random, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from robust_deconfounding.utils import  get_funcbasis
from utils_nonlinear import get_results, get_data, get_conf, bootstrap, plot_settings, double_bootstrap
from synthetic_data import functions_nonlinear

"""
    We run a simulation study to test how the confidence intervals proposed in the thesis perform, in particualr we want to investigate their coverage.
    There are 4 methods tested, the two analytical versions using the all obersvations or only the estimated inliers of the transfromed sample, bootstraping and double bootstraping.
    The experiment can take several hours to run, therefore the values are alread saved. To rerun the experiment, set the "run_exp" variable to True.
    The underlying true function can be selected over the variable "f".
"""

run_exp=False       # Set to True for running the whole experiment and False to plot an experiment which was already run
f="sigmoid"         # "sine" | "sigmoid", the underlying true function

# ----------------------------------
# parameters
# ----------------------------------

test_points=np.array([0.1, 0.5, 0.9])                           # points for which the results are plotted
alpha=np.array([0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]) # covergae to comute the confidence intervals 
noise_vars = 1                                                  # Variance of the noise
n = 2**10                                                       # number of observations n

data_args = {
    "process_type": "uniform",      # "uniform" | "oure"
    "basis_type": "cosine",         # "cosine" | "haar"
    "fraction": 0.25,               # fraction of frequencies that are confounded
    "beta": np.array([2 if f=="sine" else 3]),      
    "band": list(range(0, 50))      # list(range(0, 50)) | None
}

method_args = {
    "a": 0.7,                       # number of frequencies to remove
    "method": "torrent",            # "torrent" | "torrent_reg"
    "basis_type": "cosine_cont",    # "cosine_cont" | "cosine_disc" | "poly"
}

m = 200                                         # Number of Monte Carlo samples to draw
L = max(np.floor(1/4*n**(1/2)).astype(int),2)     # number of basis functions

colors, ibm_cb = plot_settings()                                    # import colors for plotting
path_results=os.path.join(os.path.dirname(__file__), "results/")    # Path to the results

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
                     

# ----------------------------------
# run experiments or load results
# ----------------------------------

n_x=len(test_points)            # number of test points
n_alpha=len(alpha)              # number of coverage levels to be considered
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])   # underlying true functions value
basis=get_funcbasis(x=test_points, L=L, type=method_args['basis_type'])                     # basis expansion for the test points

# Initialize storage for the obtained coverage levels
cov_l, cov_b, cov_db, cov_h=np.full([n_alpha, n_x], np.nan) , np.full([n_alpha, n_x], np.nan), np.full([n_alpha, n_x], np.nan), np.full([n_alpha, n_x], np.nan)

if run_exp:
    # Initialize arrays to compute coveragess
    T_l,T_b, T_db,T_h = np.full([n_alpha, m, n_x], np.nan), np.full([n_alpha, m, n_x], np.nan), np.full([n_alpha, m, n_x], np.nan), np.full([n_alpha, m, n_x], np.nan)
    for _ in tqdm(range(m)):
        # Generate a sample and fit DecoR
        data_values = get_data(n, **data_args, noise_var=noise_vars)
        data_values.pop('u') 
        data_values.pop('outlier_points')
        estimates_decor = get_results(**data_values, **method_args, L=L)
        y_est=basis@ estimates_decor["estimate"]

        # Bootstraping and double bootstraping
        M=100   # number of samples for the first level of the double bootstrap
        B=200   # number of samples drawn for the bootstrap

        # Perform double bootstraping to estimate the acutal coverage levels
        cov_double=double_bootstrap(x_test=test_points, transformed=estimates_decor["transformed"], estimate=estimates_decor["estimate"], a=method_args["a"], L=L, basis_type=method_args["basis_type"], M=M, B=B)
        
        # Boostrapping
        boot=bootstrap(x_test=test_points, transformed=estimates_decor["transformed"], a=method_args["a"], L=L, basis_type=method_args["basis_type"], M=B)
        n_double=len(cov_double['nominal'])

        for i in range(n_alpha):

            # Get the adjuste alpha from the double bootstrap. This is done for every point seperately since the coverage can differ.
            ci_db=np.full([n_x,2], np.nan)
            for j in range(n_x):
                ind_h=np.max(np.concatenate((np.arange(n_double)[alpha[i]>=cov_double['actual'][:,j]], [0])))
                ind_alpha=np.min([ind_h+1, int(B/2)-1])
                alpha_boot=cov_double['nominal'][ind_alpha]
                ci_db[j,:]=np.array([2*y_est[j]-boot[int(np.ceil((1+alpha_boot)/2*B)),j], 2*y_est[j]-boot[int(np.floor((1-alpha_boot)/2*B)),j]])

            # Compute the confidence intervals
            ci_h=get_conf(x=test_points, **estimates_decor, L=L, alpha=alpha[i], basis_type=method_args["basis_type"], w=0)
            ci_b=np.stack(([2*y_est-boot[int(np.ceil((1+alpha[i])/2*B)),:], 2*y_est-boot[int(np.floor((1-alpha[i])/2*B)),:]]), axis=-1)
            ci_l=get_conf(x=test_points, **estimates_decor, L=L, alpha=alpha[i], basis_type=method_args["basis_type"], w=1)

            # Check if true the function f is contained
            T_h[i,_,:]=(ci_h[:,1]>=y_true[:,0]) & (y_true[:,0]>=ci_h[:, 0])
            T_b[i,_,:]=(ci_b[:,1]>=y_true[:,0]) & (y_true[:,0]>=ci_b[:,0])
            T_db[i,_,:]=(ci_db[:,1]>=y_true[:,0]) & (y_true[:,0]>=ci_db[:,0])
            T_l[i,_,:]=(ci_l[:,1]>=y_true[:,0]) & (y_true[:,0]>=ci_l[:,0])

    # Compute the actual estimated coverage
    for i in range(n_alpha):
         cov_l[i, :], cov_b[i,:], cov_db[i,:], cov_h[i,:]=np.sum(T_l[i, :, :], axis=0)/m, np.sum(T_b[i, :, :], axis=0)/m, np.sum(T_db[i, :, :], axis=0)/m, np.sum(T_h[i,:,:], axis=0)/m

    #Save the results using a pickle file
    with open(path_results+"confidence_interval_" + str(data_args["beta"])+ '.pkl', 'wb') as fp:
            cov={'cov_l': cov_l, 'cov_b': cov_b, 'cov_db': cov_db, 'cov_h': cov_h}
            pickle.dump(cov, fp)

else:
    # Loading the file with the saved results
    with open(path_results+"confidence_interval_" + str(data_args["beta"])+'.pkl', 'rb') as fp:
        cov = pickle.load(fp)
        cov_l, cov_b, cov_db, cov_h= cov['cov_l'], cov['cov_b'], cov['cov_db'], cov['cov_h']


# ----------------------------------
# Plotting
# ----------------------------------

n_plots=len(test_points)
fig, axs = plt.subplots(1, n_plots ,figsize=(12, 5), layout='constrained')
alpha_min, alpha_max=np.min(alpha), np.max(alpha)

for i in range(0, n_plots):
    axs[i].plot(alpha, cov_l[ :,i], '--o', color=ibm_cb[1])
    axs[i].plot(alpha, cov_db[ :,i], '--o', color=ibm_cb[2])
    axs[i].plot(alpha, cov_b[ :,i], '--o', color=ibm_cb[3])
    axs[i].plot(alpha, cov_h[ :,i], '--o', color=ibm_cb[4])
    axs[i].plot([alpha_min, alpha_max], [alpha_min, alpha_max], '--', color="black")
    # Labels
    axs[i].set_xlabel('nominal coverage')
    axs[i].set_ylabel('est. actual coverage')
    axs[i].set_title('x=' + str(test_points[i]))

# Legend   
def get_handles():
    point_1 = Line2D([0], [0], label="inliers", marker='o', markersize=10,
                     color=ibm_cb[1], linestyle='--')
    point_2 = Line2D([0], [0], label="double \n bootstrap", marker='o', markersize=10,
                     color=ibm_cb[2], linestyle='--')
    point_3 = Line2D([0], [0], label="all", marker='o', markersize=10,
                     color=ibm_cb[4], linestyle='--')
    point_4 = Line2D([0], [0], label="bootstrap", marker='o', markersize=10,
                     color=ibm_cb[3], linestyle='--')
    return [ point_3, point_4, point_2, point_1]

fig.subplots_adjust(right=10)
fig.legend(handles=get_handles(), loc='outside right center')
plt.tight_layout(rect=[0, 0, 0.89, 1.0])
plt.show()