import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from utils_nonlinear import get_results, plot_results, get_data, plot_settings, get_conf
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
    "process_type": "blpnl",       # "ou" | "blp" | "blpnl"
    "basis_type": "cosine",     # "cosine" | "haar"
    "fraction": 0.25,
    "beta": np.array([3]),
    "band": list(range(0, 50))  # list(range(0, 50)) | None
}

method_args = {
    "a": 0.7,
    "method": "torrent",        # "torrent" | "bfs"
}


noise_vars =  1
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
#Generate the data
data_values = get_data(n, **data_args, noise_var=noise_vars, noise_type="normal")
#Save the outlier u and its coresponding frequency
u=data_values.pop("u")
outlier_points=data_values.pop("outlier_points")
#Estimate the function f
estimates_decor = get_results(**data_values, **method_args, L=L_temp)
ci=get_conf(x=test_points, **estimates_decor, alpha=0.95)
estimates_fourrier= get_results(**data_values, method="ols", L=L_temp, a=0)
ci_fourier=get_conf(x=test_points, **estimates_fourrier, alpha=0.95)
y_est=basis @ estimates_decor["estimate"]
y_fourrier= basis @ estimates_fourrier["estimate"]
y_est=np.ndarray((n_x, 1), buffer=y_est)
#Compute the L^2-error
print("$L^2$-error: ", 1/np.sqrt(n_x)*np.linalg.norm(y_true-y_est, ord=2))

# ----------------------------------
# plotting
# ----------------------------------

sub=np.linspace(0, n-1, 2**8).astype(int)
plt.plot(data_values['x'][sub],data_values['y'][sub], 'o:w', mec = 'black', ls="")
plt.plot(test_points, y_true, '-', color='black')
plt.plot(test_points, y_est, '-', color=ibm_cb[1])
plt.plot(test_points, y_fourrier, color=ibm_cb[4])

plt.fill_between(test_points, y1=ci[:, 0], y2=ci[:, 1], color=ibm_cb[1], alpha=0.1)
plt.fill_between(test_points, y1=ci_fourier[:, 0], y2=ci_fourier[:, 1], color=ibm_cb[4], alpha=0.1)

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

plt.legend(handles=get_handles(), loc="lower right")
plt.tight_layout()
plt.show()


#Plotting the confounder

plt.plot(data_values['x'][sub], u[sub], 'o:w', mec=ibm_cb[4] )

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

#Get the sets of outliers
inliniers=set(estimates_decor["inliers"])
true_outliers=set(outlier_points.flatten())
detected_outliers=true_outliers.difference(inliniers)
not_detected_outliers=true_outliers.difference(detected_outliers)
true_intliniers=inliniers.difference(true_outliers)

m_plots=np.ceil(L_temp/2).astype(int)
fig, axs = plt.subplots(m_plots, 2)

for l in range(0, L_temp):
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

def get_handles():

    point_1 = Line2D([0], [0], label='outliers found', marker='o', color=ibm_cb[1], ls='')
    point_2 = Line2D([0], [0], label='outliers missed', marker='o', color=ibm_cb[4], ls='')
    point_3 = Line2D([0], [0], label="inliniers found" , marker='o', mec=ibm_cb[0], color='w')

    return [point_1, point_2, point_3]

plt.suptitle("Detected Outliers")
plt.tight_layout()
fig.subplots_adjust(top=0.8)
fig.legend(handles=get_handles(), loc="upper center", bbox_to_anchor=(0.55, 0.9), ncol=3)

plt.show()