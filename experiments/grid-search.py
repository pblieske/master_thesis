import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


noise_vars =  0
n = 2 ** 8 # number of observations
print("number of observations:", n)

# ----------------------------------
# run experiments
# ----------------------------------
n_x=200
test_points=np.array([i / n_x for i in range(0, n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])

#Choose the grid
L=np.array(range(1, 50))                              #Number of coefficients used
Lmbd=np.array([2**(i/2) for i in range(-20, 20)])     #Regularization parameters

#Initialize matrix to save results
err =np.empty(shape = [L.size, Lmbd.size]) 

#Get data
data_values = get_data(n, **data_args, noise_var=noise_vars)
u=data_values["u"]
data_values.pop('u')

for l in L:
    #Compute the basis
    basis_tmp = [np.cos(np.pi * test_points * k ) for k in range(0, l)] 
    basis = np.vstack(basis_tmp).T
    diag=np.array([i**4 for i in range(0,l)])
    K=np.diag(diag)
    for j in range(0, Lmbd.size-1):
         #Estimate the function f
        estimates_decor = get_results(**data_values, **method_args, L=l, K=K, lmbd=Lmbd[j])
        y_est=basis @ estimates_decor["estimate"]
        y_est=np.ndarray((n_x, 1), buffer=y_est)
        #Compute the L^2-error
        err[l-1, j]=1/np.sqrt(n_x)*np.linalg.norm(y_true-y_est, ord=2)

print(err)

# ----------------------------------
# plotting
# ----------------------------------

plt.imshow(err) 
# Add colorbar 
plt.colorbar() 
plt.title( "L^2-error" ) 
plt.xlabel("log(\lambda)")
plt.ylabel("L")
plt.show() 
