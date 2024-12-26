import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pygam as gam
from pygam import LinearGAM, s, f

from utils_nonlinear import get_results, plot_results, get_data, plot_settings
from synthetic_data import functions_nonlinear

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
L_temp=20
print("number of coefficients:", L_temp)
#Compute the basis
basis_tmp = [np.cos(np.pi * test_points * k ) for k in range( L_temp)] 
basis = np.vstack(basis_tmp).T
#Get data
data_values = get_data(n, **data_args, noise_var=noise_vars)
data_values.pop('u')
#Estimate the function f
diag=np.concatenate((np.array([0]), np.array([i**4 for i in range(1,L_temp)])))
K=np.diag(diag)
lmbd=np.concatenate((np.array([0]), np.array([2**(i/4) for i in range(-100,  20)])))
estimates_decor = get_results(**data_values, **method_args, K=K, L=L_temp, lmbd=lmbd)
y_est=basis @ estimates_decor["estimate"]
y_est=np.ndarray((n_x, 1), buffer=y_est)
#Compute the L^2-error
print("$L^2$-error: ", 1/np.sqrt(n_x)*np.linalg.norm(y_true-y_est, ord=2))

# ----------------------------------
# plotting
# ----------------------------------

sub=np.linspace(0, n-1, 2**8).astype(int)
plt.plot(data_values['x'][sub],data_values['y'][sub], 'o:w', mec = 'black')
plt.plot(test_points, y_true, '-', color='black')
plt.plot(test_points, y_est, '-', color=ibm_cb[1])

titles = {"blp": "Band-Limited", "ou": "Ornstein-Uhlenbeck", "blpnl" : "Nonlinear: Band-Limited"}
titles_basis = {"cosine": "", "haar": ", Haar basis"}
titles_dim = {1: "", 2: ", 2-dimensional"}


def get_handles():

    point_1 = Line2D([0], [0], label='Observations', marker='o', mec='black', color='w')
    point_2 = Line2D([0], [0], label='Truth', markeredgecolor='w', color='black', linestyle='-')
    point_3 = Line2D([0], [0], label="DecoR" , color=ibm_cb[1], linestyle='-')

    return [point_1, point_2, point_3]


plt.xlabel("x")
plt.ylabel("y")
plt.title(titles[data_args["process_type"]]
          + titles_basis[data_args["basis_type"]]
          + titles_dim[len(data_args["beta"])])

plt.legend(handles=get_handles(), loc="lower left")
plt.tight_layout()
plt.show()



