import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pygam import LinearGAM, s, f

from utils_nonlinear import get_results, get_conf, get_data, plot_settings
from synthetic_data import functions_nonlinear

import sys
sys.path.insert(0, '/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/code/master_thesis')
from robust_deconfounding.utils import cosine_basis


"""
We provide a visualization of a fitted curve using the cosine approximation.
For this we simulated only one draw for a fixed number of observations n, for Monte Carlo simulation look at experiments_nonlinear.py.
"""

colors, ibm_cb = plot_settings()

SEED = 5
np.random.seed(SEED)
random.seed(SEED)

data_args = {
    "process_type": "blpnl",    # "ou" | "blp" | "blpnl"
    "basis_type": "cosine",     # "cosine" | "haar"
    "fraction": 0.25,
    "beta": np.array([2]),
    "band": list(range(0, 50))  # list(range(0, 50)) | None
}

method_args = {
    "a": 0.7,
    "method": "torrent",        # "torrent" | "bfs"
}


noise_vars =  1
n = 2 ** 8 # number of observations
print("number of observations:", n)

# ----------------------------------
# run experiments
# ----------------------------------
n_x=200
test_points=np.array([i / n_x for i in range(0, n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])
L=max((np.floor(n**(1/2)/4)).astype(int),1)
print("number of coefficients:", L)

#Compute the basis
basis_tmp = [np.cos(np.pi * test_points * k ) for k in range( L)] 
basis = np.vstack(basis_tmp).T

#Get data
data_values = get_data(n, **data_args, noise_var=noise_vars)
x=data_values['x']
y=data_values['y']
data_values.pop('u')

#Fit the DecoR estimator
estimates_decor = get_results(**data_values, **method_args, L=L)
inliers=estimates_decor["inliers"]
y_decor=basis @ estimates_decor["estimate"]
y_decor=np.ndarray((n_x, 1), buffer=y_decor)

#Remove the confounded frequencies
outliers=np.delete(np.arange(0,n), list(inliers))
basis=cosine_basis(n)
coef_conf=1/n*basis[outliers, :]@y
y_deco=y-basis[outliers, :].T@coef_conf

#Fit GAM on the deconfounded data
gam =LinearGAM(n_splines=n)
gam.gridsearch(x,y_deco, lam=np.array([10**(i/5) for i in range(-50, 50)]))
gam.summary()
y_gam=gam.predict(test_points)

#Compute the confidence intervals
conf_gam=gam.confidence_intervals(test_points, width=0.95)
conf_decor=get_conf(x=test_points, **estimates_decor, alpha=0.95)

#Compute the L^2-error
print("$L^2-error")
print("DecoR: ", 1/np.sqrt(n_x)*np.linalg.norm(y_true-y_decor, ord=2))
print("GAM: ", 1/np.sqrt(n_x)*np.linalg.norm(y_true-y_gam, ord=2))

# ----------------------------------
# plotting
# ----------------------------------

sub=np.linspace(0, n-1, 2**8).astype(int)
plt.plot(data_values['x'][sub],y_deco[sub], 'o:w', mec = 'black')
plt.plot(test_points, y_true, '-', color='black')
plt.plot(test_points, y_decor, '-', color=ibm_cb[1])
plt.plot(test_points, y_gam, color=ibm_cb[4])
plt.fill_between(test_points, y1=conf_decor[:, 0], y2=conf_decor[:, 1], color=ibm_cb[1], alpha=0.1)
plt.fill_between(test_points, y1=conf_gam[:, 0], y2=conf_gam[:, 1], color=ibm_cb[4], alpha=0.1)

def get_handles():

    point_1 = Line2D([0], [0], label='Observations', marker='o', mec='black', color='w')
    point_2 = Line2D([0], [0], label='Truth', markeredgecolor='w', color='black', linestyle='-')
    point_3 = Line2D([0], [0], label="DecoR" , color=ibm_cb[1], linestyle='-')
    point_4 = Line2D([0], [0], label="GAM" , color=ibm_cb[4], linestyle='-')
    return [point_1, point_2, point_3, point_4]


plt.xlabel("x")
plt.ylabel("y")
plt.title("GAM")

plt.legend(handles=get_handles(), loc="lower left")
plt.tight_layout()
plt.show()
