import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils_nonlinear import get_results, plot_results, get_data, plot_settings
from synthetic_data import functions_nonlinear

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
    "fraction": 0.3,
    "beta": np.array([2]),
    "band": list(range(0, 50))  # list(range(0, 50)) | None
}

method_args = {
    "a": 0.6,
    "method": "torrent",        # "torrent" | "bfs"
}


noise_vars =  0.2
n = 2 ** 10 # number of observations
print("number of observations:", n)

# ----------------------------------
# run experiments
# ----------------------------------
n_x=200
test_points=np.array([i / n_x for i in range(0, n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])
L_temp=max(np.floor(1/4*n**(1/2)).astype(int),1)                        #Number of coefficients used
print("number of coefficients:", L_temp)
#Compute the basis
basis_tmp = [np.cos(np.pi * test_points * k ) for k in range( L_temp)] 
basis = np.vstack(basis_tmp).T
#Get data
data_values = get_data(n, **data_args, noise_var=noise_vars)
u=data_values["u"]
data_values.pop('u')
#Estimate the function f
estimates_decor = get_results(**data_values, **method_args, L=L_temp)
estimates_fourrier= get_results(**data_values, method="ols", L=L_temp, a=0).T
y_est=basis @ estimates_decor
y_fourrier= basis @ estimates_fourrier
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
plt.plot(test_points, y_fourrier, color=ibm_cb[4])

titles = {"blp": "Band-Limited", "ou": "Ornstein-Uhlenbeck", "blpnl" : "Nonlinear: Band-Limited"}
titles_basis = {"cosine": "", "haar": ", Haar basis"}
titles_dim = {1: "", 2: ", 2-dimensional"}


def get_handles():

    point_1 = Line2D([0], [0], label='Observations', marker='o', mec='black', color='w')
    point_2 = Line2D([0], [0], label='Truth', markeredgecolor='w', color='black', linestyle='-')
    point_3 = Line2D([0], [0], label="DecoR" , color=ibm_cb[1], linestyle='-')
    point_4= Line2D([0], [0], label="OLS" , color=ibm_cb[4], linestyle='-')

    return [point_1, point_2, point_3, point_4]


plt.xlabel("x")
plt.ylabel("y")
plt.title(titles[data_args["process_type"]]
          + titles_basis[data_args["basis_type"]]
          + titles_dim[len(data_args["beta"])])

plt.legend(handles=get_handles(), loc="lower left")

plt.tight_layout()
plt.show()


#Plotting the confounder
plt.plot(data_values['x'][sub], u[sub], 'o:w', mec=ibm_cb[4] )

def get_handles():

    point_1 = Line2D([0], [0], label='Confounder', marker='o',color=ibm_cb[4])
    return [point_1]

plt.xlabel("x")
plt.ylabel("y")
plt.title("Confounder")

plt.legend(handles=get_handles(), loc="upper right")
plt.hlines(0, 0, 1, colors='black', linestyles='dashed')
plt.tight_layout()
plt.show()
