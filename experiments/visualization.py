import numpy as np
import random
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from robust_deconfounding.utils import get_funcbasis
from utils_nonlinear import get_results, get_data, plot_settings, get_conf
from synthetic_data import functions_nonlinear

"""
We provide a visualization of a fitted curve using the cosine approximation.
For this we simulated only one draw for a fixed number of observations n, for Monte Carlo simulations look at consistency.py.
"""

colors, ibm_cb = plot_settings()

SEED = 4
np.random.seed(SEED)
random.seed(SEED)

data_args = {
    "process_type": "oure",    #  "blpnl" | "ounl" | "unifrom" | "oure"
    "basis_type": "cosine",    # "cosine" | "haar"
    "fraction": 0.25,
    "noise_type": "normal",
    "noise_var": 1,
    "beta": np.array([2]),
    "band": list(range(0, 50)),  # list(range(0, 50)) | None
}

method_args = {
    "a": 0.7,
    "method": "torrent",        # "torrent" | "bfs"
    "basis_type": "cosine_cont",# basis used for the approximation of f, corresponding to \psi in the paper
}

benchmark="spline"  # "spline" | "ols"
n = 2**8            # number of observations
print("number of observations:", n)


# ----------------------------------
# run the experiment
# ----------------------------------

n_x=200     # Resolution of x-axis
test_points=np.array([i / n_x for i in range(n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])
L=max(np.floor(1/4*n**(1/2)).astype(int),2)    #Number of coefficients used
print("number of coefficients:", L)

#Compute the basis and generate the data
basis=get_funcbasis(x=test_points, L=L, type=method_args["basis_type"])
data_values = get_data(n, **data_args)
u=data_values.pop("u")
outlier_points=data_values.pop("outlier_points")

#Estimate the function f by DecoR
estimates_decor = get_results(**data_values, **method_args, L=L)
ci=get_conf(x=test_points, **estimates_decor, L=L, alpha=0.95, basis_type=method_args["basis_type"])
y_est=basis @ estimates_decor["estimate"]
y_est=np.ndarray((n_x, 1), buffer=y_est)

#Estimate the function by the benchmark
if benchmark=="ols":
    estimates_fourier= get_results(**data_values, method="ols", basis_type=method_args["basis_type"], L=L, a=0, outlier_points=outlier_points)
    ci_bench=get_conf(x=test_points, **estimates_fourier, alpha=0.95, basis_type=method_args["basis_type"])
    y_bench= basis @ estimates_fourier["estimate"]
elif benchmark=="spline":
    x=np.reshape(data_values["x"], (-1,1))
    y=data_values["y"]
    gam = LinearGAM(s(0)).gridsearch(x, y)
    y_bench=gam.predict(test_points)
    ci_bench=gam.confidence_intervals(test_points, width=0.95)
else:
    raise ValueError("benchmark not implemented")

#Compute the L^2-error and L^1-error
print("$L^2$-error: ", 1/np.sqrt(n_x)*np.linalg.norm(y_true-y_est, ord=2))
print("$L^1$-error: ", 1/n_x*np.linalg.norm(y_true-y_est, ord=1))

# ----------------------------------
# plotting
# ----------------------------------

# Plotting the estimated function
plt.scatter(x=data_values['x'], y=data_values['y'], marker='o', color='w', edgecolors='black') 
plt.plot(test_points, y_true, '-', color='black')
plt.plot(test_points, y_est, '-', color=ibm_cb[1])
plt.plot(test_points, y_bench, color=ibm_cb[4])

plt.fill_between(test_points, y1=ci[:, 0], y2=ci[:, 1], color=ibm_cb[1], alpha=0.1)
plt.fill_between(test_points, y1=ci_bench[:, 0], y2=ci_bench[:, 1], color=ibm_cb[4], alpha=0.1)

# Labeling
def get_handles():
    point_1 = Line2D([0], [0], label='Observations', marker='o', mec='black', color='w')
    point_2 = Line2D([0], [0], label='Truth', markeredgecolor='w', color='black', linestyle='-')
    point_3 = Line2D([0], [0], label="DecoR" , color=ibm_cb[1], linestyle='-')
    point_4= Line2D([0], [0], label="GAM" , color=ibm_cb[4], linestyle='-')

    return [point_1, point_2, point_3, point_4]

plt.xlabel("x")
plt.ylabel("y")
plt.title("Example")

plt.legend(handles=get_handles(), loc="upper right")
plt.tight_layout()
plt.show()

# ----------------------------------
# Plotting the confounder
# ----------------------------------

plt.plot(data_values['x'], u, 'o:w', mec=ibm_cb[4] )

def get_handles():
    point_1 = Line2D([0], [0], label='Confounder', marker='o', color='w', mec=ibm_cb[4], ls='')

    return [point_1]

plt.xlabel("x")
plt.ylabel("u")
plt.title("Confounder")
plt.legend(handles=get_handles(), loc="upper right")
plt.hlines(0, 0, 1, colors='black', linestyles='dashed')
plt.tight_layout()
plt.show()

# ----------------------------------
# plotting the outliers
# ----------------------------------

trans=estimates_decor["transformed"]
P_n=trans["xn"]
y_n=trans["yn"]

# Get the sets of outliers
inliniers=set(estimates_decor["inliers"])
true_outliers=outlier_points
detected_outliers=true_outliers.difference(inliniers)
not_detected_outliers=true_outliers.difference(detected_outliers)
true_intliniers=inliniers.difference(true_outliers)

m_plots=np.ceil(L/2).astype(int)
fig, axs = plt.subplots(m_plots, 2)

# Plotting the transformed sample against P(i, :)
for l in range(0, L):
    i=np.floor(l/2).astype(int)
    j=np.mod(l,2)
    axs[i, j].plot(P_n[:, l], y_n, 'o:w', mec='black')
    axs[i, j].plot(P_n[list(detected_outliers), l], y_n[list(detected_outliers)], 'o', color=ibm_cb[1])
    axs[i, j].plot(P_n[list(not_detected_outliers), l], y_n[list(not_detected_outliers)], 'o', color=ibm_cb[4])
    axs[i, j].plot(P_n[list(true_intliniers), l], y_n[list(true_intliniers)], 'o:w', mec=ibm_cb[0])
    axs[i,j].axline((0,0), slope=estimates_decor["estimate"][l],color = 'black', linestyle = '--')
    #Labels
    axs[i, j].set_xlabel('$P(' + str(l) + ', : )$')
    axs[i, j].set_ylabel('T(Y)')

# Labels
def get_handles():
    point_1 = Line2D([0], [0], label='outliers found', marker='o', color=ibm_cb[1], ls='')
    point_2 = Line2D([0], [0], label='outliers missed', marker='o', color=ibm_cb[4], ls='')
    point_3 = Line2D([0], [0], label="inliers found" , marker='o', mec=ibm_cb[0], color='w')

    return [point_1, point_2, point_3]

plt.tight_layout()
fig.subplots_adjust(top=0.8)
fig.legend(handles=get_handles(), loc="upper center", bbox_to_anchor=(0.55, 0.9), ncol=3)
fig.set_size_inches(6.8,5)

plt.show()