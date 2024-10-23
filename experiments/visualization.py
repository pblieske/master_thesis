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

SEED = 2
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

noise_vars =  0.5
n = 2 ** 8   # number of observations

# ----------------------------------
# run experiments
# ----------------------------------
n_x=100
test_points = np.array([i / n_x for i in range(1, n_x)])
y_true=(test_points -np.full((n_x, 1), 0.5, dtype=float))**2


res = {"DecoR": []}
res["DecoR"].append([])
L_temp=max(np.floor(1/4*n**(1/2)).astype(int),1)
basis_tmp = [np.cos(np.pi * test_points * k ) for k in range(L_temp)]
basis = np.vstack(basis_tmp).T
data_values = get_data(n, **data_args, noise_var=noise_vars[i])
estimates_decor = get_results(**data_values, **method_args, L=L_temp)
y_est=basis @ estimates_decor

res["DecoR"][-1].append(1/n_x*np.linalg.norm(y_true-y_est, ord=1))
""""
estimates_ols = get_results(**data_values, method="ols", a=method_args["a"])
res["ols"][-1].append(np.linalg.norm(estimates_ols - data_args["beta"].T, ord=1))
"""

res["DecoR"] = np.array(res["DecoR"])
plot_results(res, num_data, m, colors=colors[i])

# ----------------------------------
# plotting
# ----------------------------------

titles = {"blp": "Band-Limited", "ou": "Ornstein-Uhlenbeck", "blpnl" : "Nonlinear: Band-Limited"}
titles_basis = {"cosine": "", "haar": ", Haar basis"}
titles_dim = {1: "", 2: ", 2-dimensional"}


def get_handles():
    """
    point_1 = Line2D([0], [0], label='OLS', marker='o',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    """
    point_2 = Line2D([0], [0], label='DecoR', marker='X',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_3 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[0]), markersize=10,
                     color=ibm_cb[1], linestyle='-')
    point_4 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[1]), markersize=10,
                     color=ibm_cb[4], linestyle='-')
    point_5 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[2]), markersize=10,
                     color=ibm_cb[2], linestyle='-')
    return [ point_2, point_3, point_4, point_5]


plt.xlabel("number of data points")
plt.ylabel("mean absolute error")
plt.title(titles[data_args["process_type"]]
          + titles_basis[data_args["basis_type"]]
          + titles_dim[len(data_args["beta"])])
plt.xscale('log')
plt.xlim(left=num_data[0] - 2)
plt.hlines(0, num_data[0], num_data[-1], colors='black', linestyles='dashed')

plt.legend(handles=get_handles(), loc="upper right")

plt.tight_layout()
plt.show()