import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

from utils_nonlinear import get_results, plot_results, get_data, plot_settings
from synthetic_data import functions_nonlinear

"""
    We test the different cross-validation methods. For this end, we do the regularization mainly over the parameter lambda 
    and choose the number of coefficients L of the oder of the number of observations n.
"""

colors, ibm_cb = plot_settings()

SEED = 5
np.random.seed(SEED)
random.seed(SEED)

data_args = {
    "process_type": "blpnl",    # "ou" | "blp" | "blpnl"
    "basis_type": "cosine",     # "cosine" | "haar"
    "fraction": 0.2,
    "beta": np.array([3]),
    "band": list(range(0, 50))  # list(range(0, 50)) | None
}

method_args = {
    "a": 0.75,
    "method": "torrent",        # "torrent" | "bfs"
}

m = 100  #Number of repetitions for the Monte Carlo
noise_vars = 1
methods=["torrent", "torrent_cv3"]
num_data = [2 ** k for k in range(6, 13)]      # [4, 8, 10]
Lmbd=np.array([10**(i/10) for i in range(-70, 10)])

# ----------------------------------
# run experiments
# ----------------------------------
n_x=200     #Resolution of x-axis
test_points = np.array([i / n_x for i in range(0, n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])

for i in range(len(methods)):
    print("Version: ", methods[i])
    res = {"DecoR": []}
    for n in num_data:
        print("number of data points: ", n)
        res["DecoR"].append([])
        L_temp=max(np.floor(n**(1/2)).astype(int),1)      
        basis_tmp = [np.cos(np.pi * test_points * k ) for k in range(L_temp)]
        basis = np.vstack(basis_tmp).T
        print("number of coefficients: ", L_temp)
        diag=np.concatenate((np.array([0]), np.array([i**4 for i in range(1,L_temp)])))
        K=np.diag(diag)

        for _ in range(m):
            data_values = get_data(n, **data_args, noise_var=noise_vars)
            data_values.pop('u', 'basis')
            estimates_decor = get_results(**data_values, a=method_args["a"], method=methods[i], L=L_temp, lmbd=Lmbd, K=K)
            y_est=basis @ estimates_decor["estimate"]
            y_est=np.ndarray((n_x, 1), buffer=y_est)

            res["DecoR"][-1].append(1/np.sqrt(n_x)*np.linalg.norm(y_true-y_est, ord=2))
            
    res["DecoR"] = np.array(res["DecoR"])

    #Plotting using seaborn
    values =  np.expand_dims(res["DecoR"], 2).ravel()
    time = np.repeat(num_data, m )
    method = np.tile([ "DecoR"], len(values))
    df = pd.DataFrame({"value": values.astype(float),
                       "n": time.astype(float),
                       "method": method})
    sns.lineplot(data=df, x="n", y="value", hue="method", style="method",
                 markers=["X"], dashes=False, errorbar=("ci", 95), err_style="band",
                 palette=[colors[i][0]], legend=True)

# ----------------------------------
# plotting
# ----------------------------------

titles = {"blp": "Band-Limited", "ou": "Ornstein-Uhlenbeck", "blpnl" : "Nonlinear: Band-Limited"}
titles_basis = {"cosine": "", "haar": ", Haar basis"}
titles_dim = {1: "", 2: ", 2-dimensional"}


def get_handles():
    point_2 = Line2D([0], [0], label='DecoR', marker='X',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_3 = Line2D([0], [0], label="Method: " + str(methods[0]), markersize=10,
                     color=ibm_cb[1], linestyle='-')
    point_4 = Line2D([0], [0], label="Method: " + str(methods[1]), markersize=10,
                     color=ibm_cb[4], linestyle='-')
    return [ point_2, point_3, point_4]


plt.xlabel("number of data points")
plt.ylabel("L^2 error")
plt.title(titles[data_args["process_type"]]
          + titles_basis[data_args["basis_type"]]
          + titles_dim[len(data_args["beta"])])
plt.xscale('log')
plt.xlim(left=num_data[0] - 2)
plt.hlines(0, num_data[0], num_data[-1], colors='black', linestyles='dashed')
plt.legend(handles=get_handles(), loc="lower left")
plt.tight_layout()

plt.show()