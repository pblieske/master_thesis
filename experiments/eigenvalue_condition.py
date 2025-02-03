import numpy as np
import random, os, pickle
import matplotlib.pyplot as plt
from utils_nonlinear import get_results, get_data, plot_settings, check_eigen

"""
    We check the eigenvalue condition of the consistency theorem numerical.
    For this, we use a Monte-Carlo simulation and plot the disribution afterwards using in a histogramm.
"""

run_exp=False        # Set to True for running the whole experiment and False to plot an experiment which was already run
n = 2 ** 10          # number of observations 
noise_var= 0        # variance of the noise


# ----------------------------------
# Parameters
# ----------------------------------

m=1000      # number of Monte-Carlos drawn  
L=max(np.floor(1/(1 if noise_var==0 else 5)**n**0.5).astype(int),1)     # number of coefficients used

data_args = {
    "process_type": "uniform",      # "uniform" | "oure"
    "basis_type": "cosine",         # "cosine" | "haar"
    "fraction": 0.25,               # fraction of frequencies that are confounded
    "beta": np.array([2]),
    "band": list(range(0, 50))      # list(range(0, 50)) | None
}

method_args = {
    "a": 0.7,                       # number of frequencies to remove
    "method": "torrent",            # "torrent" | "torrent_reg"
    "basis_type": "cosine_cont",    # "cosine_cont" | "cosine_disc" | "poly"
}

T=np.full(m, np.nan)                # Allocate memory the save the results
colors, ibm_cb = plot_settings()    # import color setting for plotting
path_results=os.path.join(os.path.dirname(__file__), "results/")    # Path to the results


# ----------------------------------
# run experiment
# ----------------------------------

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

if run_exp:
    for i in range(m):
        #Generate the data and run DecoR
        data_values = get_data(n, **data_args, noise_var=noise_var)
        u=data_values.pop("u")
        outlier_points=data_values.pop("outlier_points")
        estimates_decor = get_results(**data_values, **method_args, L=L)
        #Check the eigenvalue condition
        T[i]=check_eigen(P=estimates_decor["transformed"]["xn"], S=estimates_decor["inliers"], G=outlier_points)["fraction"]

    # Save the results
    with open(path_results+"eignevalue_cond_sigma="+str(noise_var)+"_n_" + str(n)+'.pkl', 'wb') as fp:
        pickle.dump(T, fp)
        print('Results saved successfully to file.')
else:
    # Loading the file with the saved results
    with open(path_results+"eignevalue_cond_sigma="+str(noise_var)+"_n_" + str(n)+'.pkl', 'rb') as fp:
        T = pickle.load(fp)


# ----------------------------------
# Plotting the histogramm
# ----------------------------------

n_bin=20        # number of bins
max=np.max(T)   
min=np.min(T)
delta=(max-min)/n_bin                                       # width of bin
x_0=1/np.sqrt(2)-np.ceil((1/np.sqrt(2)-min)/delta)*delta    # strating value for the first bin
bins=np.array([i*delta+x_0 for i in range(-1,n_bin+1)])     # grid for the histogramm

#Labeling
plt.hist(T, bins=bins, color=ibm_cb[0], edgecolor='k', alpha=0.6)
plt.axvline(1/np.sqrt(2), color=ibm_cb[2])
plt.xlabel("fraction")
plt.ylabel("count")
plt.title("Eigenvalue Condition: " + "$n="+ str(n) + "$ and $ \sigma_{\eta}^2=" + str(noise_var) + "$")
plt.text(1/np.sqrt(2)+0.1, 400,"$1/\sqrt{2}$", color=ibm_cb[2], rotation=90, fontdict={'fontsize': 12, 'fontweight': 'bold'})
plt.tight_layout()
plt.show()