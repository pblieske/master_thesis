import numpy as np
import random, tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from robust_deconfounding.utils import  get_funcbasis
from utils_nonlinear import get_data, plot_settings, get_results, err_boot
from synthetic_data import functions_nonlinear

"""
    We plot the regularization path including the standard error of the three out-of-boostrap gernerlization error estimation methods on the transformed sample. 
    The three methods are clipping, taking the smallest share and take the median of the residuals.
"""

# ----------------------------------
# Parameters
# ----------------------------------

B=500                   # Number of bootsrap samples to draw
Lmbd_min=10**(-8)       # smallest regularization parameter lambda to be considered
Lmbd_max=10**(1)        # largest regularization paramter lambda to be considered
n_lmbd=200              # number of lambda to test
L=30                    # number of coefficient for the reuglarized torrent
                                                                           
Lmbd=np.array([np.exp(i/n_lmbd*(np.log(Lmbd_max)-np.log(Lmbd_min))+np.log(Lmbd_min)) for i in range(0, n_lmbd)])    # grid of regularization paramters   
noise_vars = 4          # Variance of the noise
n = 2**8                # number of observations n

data_args = {
    "process_type": "uniform",      # "uniform" | "oure"
    "basis_type": "cosine",         # "cosine" | "haar"
    "fraction": 0.25,               # fraction of frequencies that are confounded
    "beta": np.array([2]),      
    "band": list(range(0, 50))      # list(range(0, 50)) | None
}

method_args = {
    "a": 0.7,                       # number of frequencies to remove
    "method": "torrent_reg",        # "torrent" | "torrent_reg"
    "basis_type": "cosine_cont",    # "cosine_cont" | "cosine_disc" | "poly"
}

colors, ibm_cb = plot_settings()    # import colors for plotting

print("number of observations:", n)
print("number of coefficients:", L)


# ----------------------------------
# Run the experiment
# ----------------------------------

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

n_x=200
test_points=np.array([i / n_x for i in range(0, n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])

#Get the data
data_values = get_data(n, **data_args, noise_var=noise_vars)
u=data_values.pop('u')
outlier_points=data_values.pop('outlier_points')

#Set up the smothness penalty
diag=np.concatenate((np.array([0]), np.array([i**4 for i in range(1,L+1)])))
K=np.diag(diag)
R=get_funcbasis(x=data_values["x"], L=L, type=method_args["basis_type"])
xn = data_values["basis"].T @ R / n
yn = data_values["basis"].T @ data_values['y'] / n

# Allocate memory
err_cap, err_inl, err_m=np.full([n_lmbd, B], float(0)), np.full([n_lmbd, B], float(0)) ,np.full([n_lmbd, B], float(0))
estimates_decor = get_results( **data_values, **method_args, L=L, lmbd=0, K=K)
basis=get_funcbasis(x=test_points, L=L, type=method_args["basis_type"])
err_true=np.full([n_lmbd], np.nan)

# Draw B boostrap samples
for i in tqdm.tqdm(range(n_lmbd)):
    boot=err_boot(transformed=estimates_decor['transformed'], a=method_args['a'], lmbd=Lmbd[i], K=K, B=B)
    err_cap[i,:], err_inl[i,:], err_m[i,:]=boot["err_cap"], boot["err_inl"], boot["err_m"]
    estimates_reg= get_results(**data_values, **method_args, L=L, K=K, lmbd=Lmbd[i])
    y_est=np.ndarray((n_x,1), buffer=basis @ estimates_reg["estimate"])
    err_true[i]=1/n_x*np.linalg.norm(y_true-y_est, ord=1)

# Compute the average estimated error
err_est=np.array([np.mean(err_cap, axis=1), np.mean(err_inl, axis=1), np.mean(err_m, axis=1)])

# Compute the standard deviation
err_sd=np.array([np.linalg.norm(err_cap-np.repeat(err_est[0].reshape(-1,1), B, axis=1), axis=1, ord=2)/(np.sqrt(B-1)), np.linalg.norm(err_inl-np.repeat(err_est[1].reshape(-1,1), B, axis=1), axis=1, ord=2)/(np.sqrt(B-1)), np.linalg.norm(err_m-np.repeat(err_est[2].reshape(-1,1), B, axis=1), axis=1, ord=2)/(np.sqrt(B-1))])

# Compute the indices minimizing the estimated perdiction error
indx_cap, indx_inl, indx_m=np.argmin(err_est[0]), np.argmin(err_est[1]), np.argmin(err_est[2])
lmbd_min=np.array([Lmbd[indx_cap], Lmbd[indx_inl], Lmbd[indx_m]])
lmbd_se=np.array([Lmbd[min(np.arange(indx_cap,n_lmbd)[err_est[0][indx_cap:n_lmbd]>err_est[0][indx_cap]+err_sd[0][indx_cap]])], Lmbd[min(np.arange(indx_inl,n_lmbd)[err_est[1][indx_inl:n_lmbd]>err_est[1][indx_inl]+err_sd[1][indx_inl]])], Lmbd[min(np.arange(indx_m,n_lmbd)[err_est[2][indx_m:n_lmbd]>err_est[2][indx_m]+err_sd[2][indx_m]])]])

# ----------------------------------
# plotting
# ----------------------------------

fig, axs = plt.subplots(1, 3, figsize=(12, 5), layout='constrained')
titles=np.array(["Clipping", "Omitting", "Median"])     # Titles of the subplots

# Plot the true error and scale the axis
for i in range(3):

    # Adjust the axis
    axs[i].set_xscale('log')
    axs[i].set_xlabel("$\lambda$")
    axs[i].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))

    # Plot the estimated generalization error
    axs[i].plot(Lmbd, err_est[i], color=ibm_cb[1])
    axs[i].fill_between(Lmbd, y1=err_est[i]-err_sd[i], y2=err_est[i]+err_sd[i], color=ibm_cb[1], alpha=0.1)
    axs[i].axvline(x=lmbd_min[i], linestyle="dashed", color=ibm_cb[2])
    axs[i].text(lmbd_min[i]/5, 0.5*min(err_est[i])+0.5*max(err_est[i]), "$\lambda_{min}$", color=ibm_cb[2], rotation=90)
    axs[i].axvline(x=lmbd_se[i], linestyle="dashed", color=ibm_cb[3])
    axs[i].text(lmbd_se[i]/5, 0.5*min(err_est[i])+0.5*max(err_est[i]), "$\lambda_{1-SE}$", color=ibm_cb[3], rotation=90)

    # Plot the true underlying error
    if i!=1:
        axs[i].plot(Lmbd, err_true/10000+0.00015, color='black', alpha=0.65)
        axs[i].set_ylim([0.00018, 0.00055])
    else:
        axs[i].plot(Lmbd, err_true/15000+0.0001, color='black', alpha=0.65)
        axs[i].set_ylim([0.00012, 0.0004])
    
    axs[i].set_title(titles[i])

axs[0].set_ylabel("Estimated Generalization Error")

# Legend   
def get_handles():
    point_1 = Line2D([0], [0], label="estimated", markersize=10,
                     color=ibm_cb[1], linestyle='-')
    point_2 = Line2D([0], [0], label="std. error", markersize=0, linewidth=10,
                     color=ibm_cb[1], linestyle='-', alpha=0.1)   
    point_3 = Line2D([0], [0], label="true loss", markersize=10,
                     color="black", linestyle='-', alpha=0.65)
    return [point_1, point_2, point_3]

fig.subplots_adjust(right=10)
fig.legend(handles=get_handles(), loc='outside right center', handlelength=2)
plt.tight_layout(rect=[0, 0, 0.88, 1.0])
plt.show()