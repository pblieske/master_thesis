import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

sys.path.insert(0, '/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/code/master_thesis')
from robust_deconfounding.utils import get_funcbasis
from utils_nonlinear import get_results, get_data, plot_settings, get_conf, check_eigen

"""
We provide a visualization of a fitted curve using the cosine approximation.
For this we simulated only one draw for a fixed number of observations n, for Monte Carlo simulations look at consistency.py.
"""

colors, ibm_cb = plot_settings()

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

n = 2 ** 12      # number of observations
m=1000          
L=max(np.floor(1/4*n**(1/2)).astype(int),1)    #Number of coefficients used


data_args = {
    "process_type": "blpnl",    # "ou" | "blp" | "blpnl" |ounl
    "basis_type": "cosine",     # "cosine" | "haar"
    "fraction": min((n**(-0.5)), 1),
    "noise_type": "normal",
    "beta": np.array([2]),
    "band": list(range(0, 50)),  # list(range(0, 50)) | None
    "noise_var": 1,
}

method_args = {
    "a": 1-data_args["fraction"]*1.25,
    "method": "torrent",        # "torrent" | "bfs"
    "basis_type": "cosine_cont",# basis used for the approximation of f
}

print("number of observations:", n)
print("number of coefficients:", L)
T=np.full(m, np.nan)

# ----------------------------------
# run experiment
# ----------------------------------
for i in range(m):
    #Generate the data and run DecoR
    data_values = get_data(n, **data_args)
    u=data_values.pop("u")
    outlier_points=data_values.pop("outlier_points")  
    diag=np.concatenate((np.array([0]), np.array([i**4 for i in range(1, L)])))
    K=np.diag(diag)
    lmbd=0
    estimates_decor = get_results(**data_values, **method_args, L=L, lmbd=lmbd, K=K)
    #Check the eigenvalue condition
    T[i]=check_eigen(x=estimates_decor["transformed"]["xn"], S=estimates_decor["inliers"], G=outlier_points, lmbd=lmbd, K=K)["fraction"]

# ----------------------------------
# plotting
# ----------------------------------

n_bin=20
max=np.max(T)
min=np.min(T)
delta=(max-min)/n_bin
x_0=1/np.sqrt(2)-np.ceil((1/np.sqrt(2)-min)/delta)*delta
bins=np.array([i*delta+x_0 for i in range(-1,n_bin+1)])

plt.hist(T, bins=bins, color=ibm_cb[0], edgecolor='k', alpha=0.6)
plt.axvline(1/np.sqrt(2), color=ibm_cb[2])
plt.xlabel("fraction")
plt.ylabel("count")
plt.title("$n="+ str(n) + "$ and $ c_n/n=" + str(data_args["fraction"]) + "$")
plt.annotate('$1/\sqrt{2}$', xy=(1/np.sqrt(2), 0), xytext=(1/np.sqrt(2)-0.5, -30), color=ibm_cb[2], arrowprops={'arrowstyle': '-'})
plt.tight_layout()
plt.show()