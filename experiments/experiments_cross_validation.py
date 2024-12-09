import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

from utils_nonlinear import get_results, get_data, plot_settings
from synthetic_data import functions_nonlinear

"""
    We test the cross-validation method. For this end, we do the regularization mainly over the parameter lambda 
    and choose the number of coefficients L of the oder of the number of observations n.
    Note that the script can take up to 3 hours to run for m=100.
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
    "method": "torrent",        # "torrent" | "bfs"
}

m = 100  #Number of repetitions for the Monte Carlo
noise_vars = [0, 0.5, 1]
num_data = [2 ** k for k in range(6, 13)]      # (6,13)
Lmbd=np.array([10**(i/10) for i in range(-100, 10)])

# ----------------------------------
# run experiments
# ----------------------------------
n_x=200     #Resolution of x-axis
test_points = np.array([i / n_x for i in range(0, n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])

for i in range(len(noise_vars)):
    print("Noise Variance: ", noise_vars[i])
    res = {"torrent": [], "cv": []}

    for n in num_data:
        print("number of data points: ", n)
        res["torrent"].append([])
        res["cv"].append([])
        #Get number of coefficients L
        L_temp=max(np.floor(n**(1/2)).astype(int),1)      
        basis_tmp = [np.cos(np.pi * test_points * k ) for k in range(L_temp)]
        basis = np.vstack(basis_tmp).T
        print("number of coefficients: ", L_temp)
        #Construct smothness penalty
        diag=np.concatenate((np.array([0]), np.array([i**4 for i in range(1,L_temp)])))
        K=np.diag(diag)
        #Run Monte Carlo simulation
        for _ in range(m):
            data_values = get_data(n, **data_args, noise_var=noise_vars[i])
            data_values.pop('u', 'basis')

            estimates_tor = get_results(**data_values, a=method_args["a"], method="torrent", L=L_temp, lmbd=0, K=K)
            estimates_cv = get_results(**data_values, a=method_args["a"], method="torrent_cv", L=L_temp, lmbd=Lmbd, K=K)
            y_tor=basis @ estimates_tor["estimate"]
            y_tor=np.ndarray((n_x, 1), buffer=y_tor)
            y_cv= basis @ estimates_cv["estimate"]
            y_cv=np.ndarray((n_x, 1), buffer=y_cv)

            res["torrent"][-1].append(1/np.sqrt(n_x)*np.linalg.norm(y_true-y_tor, ord=2))
            res["cv"][-1].append(1/np.sqrt(n_x)*np.linalg.norm(y_true-y_cv, ord=2))
            
    res["torrent"], res["cv"] = np.array(res["torrent"]), np.array(res["cv"])

    #Plotting using seaborn
    values = np.concatenate([np.expand_dims(res["torrent"], 2),
                             np.expand_dims(res["cv"], 2)], axis=2).ravel()
    time = np.repeat(num_data, m * 2)
    method = np.tile(["Torrent", "CV"], len(values) // 2)
    df = pd.DataFrame({"value": values.astype(float),
                       "n": time.astype(float),
                       "method": method})
    sns.lineplot(data=df, x="n", y="value", hue="method", style="method",
                 markers=["o", "X"], dashes=False, errorbar=("ci", 95), err_style="band",
                 palette=colors[i], legend=True)


# ----------------------------------
# plotting
# ----------------------------------

titles = {"blp": "Band-Limited", "ou": "Ornstein-Uhlenbeck", "blpnl" : "Nonlinear: Band-Limited"}
titles_basis = {"cosine": "", "haar": ", Haar basis"}
titles_dim = {1: "", 2: ", 2-dimensional"}


def get_handles():
    point_1 = Line2D([0], [0], label='Torrent', marker='o',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_2 = Line2D([0], [0], label='CV', marker='X',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_3 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[0]), markersize=10,
                     color=ibm_cb[1], linestyle='-')
    point_4 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[1]), markersize=10,
                     color=ibm_cb[4], linestyle='-')
    point_5 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[2]), markersize=10,
                     color=ibm_cb[2], linestyle='-')
    return [point_1, point_2, point_3, point_4, point_5]


plt.xlabel("number of data points")
plt.ylabel("L^2 error")
plt.title("Regularization with Cross-Validation")
plt.xscale('log')
plt.xlim(left=num_data[0] - 2)
plt.hlines(0, num_data[0], num_data[-1], colors='black', linestyles='dashed')
plt.legend(handles=get_handles(), loc="lower left")
plt.tight_layout()

plt.show()