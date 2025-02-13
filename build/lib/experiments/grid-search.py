import random, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils_nonlinear import get_results, get_data, plot_settings
from synthetic_data import functions_nonlinear
from robust_deconfounding.utils import get_funcbasis


"""
    We compute the L^1-error varing the number of coefficients and the regularization parameter \lambda. 
    For the regularization the smoothness penalty is used.
    To plot the data from an experiment which was already run, set the "run_exp" variable to False.
    The experiments were aleardy run with the following paramter configurations:
    - nois_vars=0, n=2**6
    - nois_vars=0, n=2**8
    - nois_vars=4, n=2**6
    - nois_vars=4, n=2**8
"""

# ----------------------------------
# Parameters varied in the thesis
# ----------------------------------

run_exp=False       # Set to True for running the whole experiment and False to plot an experiment which was already run
noise_vars = 0      # variance of the noise
n = 2 ** 8          # number of observations


# ----------------------------------
# Parameters kept constant
# ----------------------------------

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

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

Lmbd_min=10**(-8)       # smallest regularization parameter lambda to be considered
Lmbd_max=10**(1)        # largest regularization paramter lambda to be considered
n_lmbd=60               # number of lambda to test
L_max=60                # largest number of basis functions to consider
m=200                   # Number of Monte Carlo samples to draw

colors, ibm_cb = plot_settings()                                        # Import color settings for plotting 
path_results=os.path.join(os.path.dirname(__file__), "results/")        # Path to the results


# ----------------------------------
# Run the experiment
# ----------------------------------

n_x=200                 # resolution of x-axis
test_points=np.array([i / n_x for i in range(0, n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])   # Compute ture underlying function

# Initialize grid and matrix to save results
L=np.array(range(1, L_max))                                                                                             # grid of number of coefficients
Lmbd=np.array([np.exp(i/n_lmbd*(np.log(Lmbd_max)-np.log(Lmbd_min))+np.log(Lmbd_min)) for i in range(0, n_lmbd)])        # grid of regularization paramters
err =np.zeros(shape = [L.size, Lmbd.size])                                                                              # matrix the save the error

if run_exp:
    # Running the Monte Carlo simulation
    for __ in range(0,m):
        data_values = get_data(n, **data_args, noise_var=noise_vars)
        data_values.pop('u')
        for l in L:
            # Compute the basis and regularization matrix K
            basis=get_funcbasis(x=test_points, L=l, type=method_args["basis_type"])
            diag=np.concatenate(([0], np.array([ i**4 for i in range(0,l)])))
            K=np.diag(diag)
            for j in range(0, Lmbd.size):
                # Estimate the function f by DecoR using l basis functions and the regularization parameter Lmbd[j]
                estimates_decor = get_results(**data_values, **method_args, L=l, K=K, lmbd=Lmbd[j])
                y_est=basis @ estimates_decor["estimate"]
                y_est=np.ndarray((n_x, 1), buffer=y_est)
                # Compute the L^1-error
                err[l-1, j]=err[l-1, j]+ 1/(m*n_x)*np.linalg.norm(y_true-y_est, ord=1)
        if __ % 10 ==0:
            print("Number of samples drawn: " + str(__))

    # Saving the results
    with open(path_results+"girdsearch_n=" + str(n) +'_noise='+str(noise_vars)+'.pkl', 'wb') as fp:
        pickle.dump(err, fp)
        print('Results saved successfully to file.')

else:
    # Loading the file with the saved results
    with open(path_results+"girdsearch_n=" + str(n) +'_noise='+str(noise_vars)+'.pkl', 'rb') as fp:
        err = pickle.load(fp)

# ----------------------------------
# plotting
# ----------------------------------

custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ibm_cb[0:5])  # Create custom colormap
X, Y = np.meshgrid(Lmbd, L)
plt.pcolormesh(X, Y, err, cmap=custom_cmap, shading='nearest', linewidth=0, rasterized=False)

# Add colorbar and labeling
plt.xscale('log')
plt.colorbar() 
plt.title('$L^1$-error') 
plt.xlabel('$\lambda$')
plt.ylabel('L')
plt.tight_layout()
plt.show()