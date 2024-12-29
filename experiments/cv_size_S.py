import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

import sys
sys.path.insert(0, '/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/code/master_thesis')
from robust_deconfounding.utils import get_funcbasis
from utils_nonlinear import get_results, get_data, plot_settings
from synthetic_data import functions_nonlinear

"""
    This is a supplementary experiment to the cross validation study inverstigating the size of the the stable set S of iliers.
"""

colors, ibm_cb = plot_settings()

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

data_args = {
    "process_type": "blpnl",    # "ou" | "blp" | "blpnl"
    "basis_type": "cosine",     # "cosine" | "haar"
    "fraction": 0.3,
    "beta": np.array([2]),
    "band": list(range(0, 50)),  # list(range(0, 50)) | None
    "noise_type": "normal"
}

method_args = {
    "a": 0.65,
    "method": "torrent",        # "torrent" | "bfs"
    "basis_type": "cosine_cont"
}

m = 200  #Number of repetitions for the Monte Carlo
noise_vars = 0.5
num_data = [4 ** k for k in range(3, 7)]      # (6,13)
Lmbd=np.array([10**(i/10) for i in range(-100, 10)])

# ----------------------------------
# run experiments
# ----------------------------------
n_x=200     #Resolution of x-axis
test_points = np.array([i / n_x for i in range(0, n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])
size=np.zeros(shape = [len(num_data), m ]) 

for i in range(0, len(num_data)):
    n=num_data[i]
    print("number of data points: ", n)
    #Get number of coefficients L
    L_temp=max(np.floor(n**(1/2)).astype(int),1)      
    basis_tmp = [np.cos(np.pi * test_points * k ) for k in range(L_temp)]
    basis = np.vstack(basis_tmp).T
    bais=get_funcbasis(x=test_points, L=L_temp, type=method_args["basis_tmp"])
    print("number of coefficients: ", L_temp)
    #Construct smothness penalty
    diag=np.concatenate((np.array([0]), np.array([i**4 for i in range(1,L_temp)])))
    K=np.diag(diag)
    #Run Monte Carlo simulation
    for k in range(m):
        data_values = get_data(n, **data_args, noise_var=noise_vars)
        data_values.pop('u', 'basis')
        S=set(np.arange(0,n))  
        for j in range(0, len(Lmbd)):
            estimates_tor = get_results(**data_values, a=method_args["a"], method="torrent_reg", L=L_temp, lmbd=Lmbd[j], K=K)
            S=S.intersection(estimates_tor["inliniers"])            
        size[i, k]=len(S)

# ----------------------------------
# plotting
# ----------------------------------

fig, axs = plt.subplots(2, 2)

axs[0, 0].hist(size[0,:],  color=ibm_cb[0], edgecolor='k', alpha=0.6)
axs[0, 0].set_title(str(num_data[0])+ " Observations")
axs[0, 1].hist(size[1,:],  color=ibm_cb[0], edgecolor='k', alpha=0.6)
axs[0, 1].set_title(str(num_data[1])+ " Observations")
axs[1, 0].hist(size[2,:],  color=ibm_cb[0], edgecolor='k', alpha=0.6)
axs[1, 0].set_title(str(num_data[2])+ " Observations")
axs[1, 1].hist(size[3,:],  color=ibm_cb[0], edgecolor='k', alpha=0.6)
axs[1, 1].set_title(str(num_data[3])+ " Observations")
#Labels
for i in range(0,2):
    for j in range(0,2):
        axs[i, j].set_xlabel('$|S|$')
        axs[i, j].set_ylabel('Count')

plt.suptitle("Size of the stable inliers set S")
plt.tight_layout()
plt.show()