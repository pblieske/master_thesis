import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils_nonlinear import get_results, plot_results, get_data, plot_settings

"""
We provide a visualization of a fitted curve using the cosine approximation.
For this we simulated only one draw for a fixed number of observations n, for Monte Carlo simulation look at experiments_nonlinear.py.
"""

colors, ibm_cb = plot_settings()

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

data_args = {
    "process_type": "blpnl",       # "ou" | "blp" | "blpnl"
    "basis_type": "cosine",     # "cosine" | "haar"
    "fraction": 0.25,
    "beta": np.array([1]),
    "band": list(range(0, 5))  # list(range(0, 50)) | None
}

method_args = {
    "a": 0.7,
    "method": "torrent",        # "torrent" | "bfs"
}

noise_vars =  0
n = 2 ** 13  # number of observations
print("number of observations:", n)

# ----------------------------------
# run experiments
# ----------------------------------
n_x=200
test_points = np.array([i / n_x for i in range(0, n_x)])
y_true=4*(test_points - np.full(n_x, 0.5, dtype=float))**2
L_temp=max(np.floor(1/4*n**(1/2)).astype(int),1)
print("number of coefficients:", L_temp)
basis_tmp = [np.cos(np.pi * test_points * k ) for k in range(L_temp)]
basis = np.vstack(basis_tmp).T
data_values = get_data(n, **data_args, noise_var=noise_vars)
estimates_decor = get_results(**data_values, **method_args, L=L_temp)
y_est=basis @ estimates_decor

# ----------------------------------
# plotting
# ----------------------------------

sub=np.linspace(0, n-1, 2**7).astype(int)
plt.plot(data_values['x'][sub],data_values['y'][sub], 'o:w', mec = 'black')
plt.plot(test_points, y_true, '-', color=colors[0][0])
plt.plot(test_points, y_est, '-', color=colors[1][1])

titles = {"blp": "Band-Limited", "ou": "Ornstein-Uhlenbeck", "blpnl" : "Nonlinear: Band-Limited"}
titles_basis = {"cosine": "", "haar": ", Haar basis"}
titles_dim = {1: "", 2: ", 2-dimensional"}


def get_handles():

    point_1 = Line2D([0], [0], label='Observations', marker='o', mec='black')

    point_2 = Line2D([0], [0], label='Truth', markeredgecolor='w', color=colors[0][0], linestyle='-')
    point_3 = Line2D([0], [0], label="Estimate" , color=colors[1][1], linestyle='-')

    return [point_1, point_2, point_3]


plt.xlabel("x")
plt.ylabel("y")
plt.title(titles[data_args["process_type"]]
          + titles_basis[data_args["basis_type"]]
          + titles_dim[len(data_args["beta"])])

plt.legend(handles=get_handles(), loc="upper left")

plt.tight_layout()
plt.show()