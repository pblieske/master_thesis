import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from utils_nonlinear import get_results, plot_results, get_data, plot_settings
from synthetic_data import functions_nonlinear


"""
    We perform a grid search varing the number of coefficients and the regularization parameter \lambda. 
    Note that in this example, we do not use any sort of cross-validation to choose lambda, but keep it fixed throughout all iterations.
"""

colors, ibm_cb = plot_settings()

SEED = 5
np.random.seed(SEED)
random.seed(SEED)

data_args = {
    "process_type": "blpnl",       # "ou" | "blp" | "blpnl"
    "basis_type": "cosine",     # "cosine" | "haar"
    "fraction": 0.3,
    "beta": np.array([2]),
    "band": list(range(0, 50))  # list(range(0, 50)) | None
}

method_args = {
    "a": 0.65,
    "method": "torrent_reg",        # "torrent" | "bfs"
}


noise_vars =  0.5
n = 2 ** 10 # number of observations
print("number of observations:", n)

# ----------------------------------
# run experiments
# ----------------------------------
n_x=200
test_points=np.array([i / n_x for i in range(0, n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])
m=200       #Number of Monte Carlo samples drwan

#Choose the grid
Lmbd_min=-2
Lmbd_max=0
L_max=50
L=np.array(range(1, L_max))                              #Number of coefficients used
Lmbd=np.array([np.exp(1)**(i/2) for i in range(Lmbd_min*20, Lmbd_max*20)])     #Regularization parameters

#Initialize matrix to save results
err =np.zeros(shape = [L.size, Lmbd.size]) 

for __ in range(0,m):#Get data
    data_values = get_data(n, **data_args, noise_var=noise_vars)
    data_values.pop('u')
    for l in L:
        #Compute the basis
        basis_tmp = [np.cos(np.pi * test_points * k ) for k in range(0, l)] 
        basis = np.vstack(basis_tmp).T
        diag=np.array([i**4 for i in range(0,l)])
        K=np.diag(diag)
        for j in range(0, Lmbd.size):
            #Estimate the function f
            estimates_decor = get_results(**data_values, **method_args, L=l, K=K, lmbd=Lmbd[j])
            y_est=basis @ estimates_decor["estimate"]
            y_est=np.ndarray((n_x, 1), buffer=y_est)
            #Compute the L^2-error
            err[l-1, j]=err[l-1, j]+ 1/(m*np.sqrt(n_x))*np.linalg.norm(y_true-y_est, ord=2)
    if __ % 10 ==0:
     print("Number of samples drawn: " + str(__))

# ----------------------------------
# plotting
# ----------------------------------

# Create a custom colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ibm_cb[0:5])  # Create custom colormap

# Adjust color limits
vmin = err.min().min()  # Minimum value in the dataset
vmax = err.max().max()  # Maximum value in the dataset


#magmaBig = cm.get_cmap('magma', 512)
#newcmp =ListedColormap(magmaBig(np.linspace(0, 0.75, 384)))
plt.imshow(err, cmap=custom_cmap, vmin=vmin, vmax=vmax)
#plt.imshow(err, aspect='0.6', cmap=newcmp)
# Add colorbar 
plt.colorbar() 
plt.title(r'$L^2$-error') 
plt.xlabel(r'$\log(\lambda)$')
plt.xticks(np.arange(0, Lmbd_max*20-Lmbd_min*20, step=10), labels=[str(5*i+Lmbd_min*10) for i in range(0, 2*(Lmbd_max-Lmbd_min))])
plt.ylabel(r'L', rotation=0)
plt.show()