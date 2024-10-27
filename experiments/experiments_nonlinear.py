import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils_nonlinear import get_results, plot_results, get_data, plot_settings
from synthetic_data import functions_nonlinear

"""
Short explanation of the variables and what they do:

Data generation:
The "process_type" can either be an Ornstein-Uhlenbeck process or a band-limited process.
The "basis_type" is the basis for which the confounder is sparse. This is also the basis used by DecoR.
The "fraction" variable is the fraction of outliers. For example "0.25" means that a fourth of the datapoints 
is confounded in the "basis"-domain.
The "beta" variable is the $\beta$ value i.e. the true causal effect. It can be two- or one-dimensional.
The "noise_var" is the variance of the noise i.e. $\sigma_{\eta}^2$.
The "band" variable is the indices of the band for the band-limited process. Does nothing if "ou" is selected for
the "process_type".

Algorithm:
The "a" variable is the upper bound for the fraction of inliers in the data.
The "method" variable can either be "torrent" or "bfs" the two robust-regression algorithms implemented. DecoR 
can be easily extended to include other robust regression techniques.

Experiments:
The "m" is the number of times we resample the data to get confidence intervals.
The "num_data" variable is a list of increasing natural numbers that indicate the amount of data tested on.
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
    "band": list(range(0, 10))  # list(range(0, 50)) | None
}

method_args = {
    "a": 0.7,
    "method": "torrent",        # "torrent" | "bfs"
}

m = 100   #Number of repetitions for the Monte Carlo
noise_vars = [0, 0.2, 0.5]
num_data = [4 * 2 ** k for k in range(1, 11)]      # [4, 8, 10]

# ----------------------------------
# run experiments
# ----------------------------------
n_x=100
test_points = np.array([i / n_x for i in range(1, n_x)])
y_true=functions_nonlinear(test_points, data_args["beta"][0])

for i in range(len(noise_vars)):
    print("Noise Variance: ", noise_vars[i])
    res = {"DecoR": []}
    for n in num_data:
        print("number of data points: ", n)
        res["DecoR"].append([])
        L_temp=max(np.floor(1/2*n**(1/2)).astype(int),1)
        basis_tmp = [np.cos(np.pi * test_points * k ) for k in range(L_temp)]
        basis = np.vstack(basis_tmp).T
        print("number of coefficients: ", L_temp)

        for _ in range(m):
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