import numpy as np
import random
import matplotlib.pyplot as plt

from robust_deconfounding.utils import  get_funcbasis
from utils_nonlinear import get_data, plot_settings, get_results, err_boot
from synthetic_data import functions_nonlinear

"""
    We plot the regularization path including the standard deviation of the three out-of-boostrap gernerlization error on the transformed sample. 
    The three methods are clipping, taking the smallest share and take the median of the residuals.
"""

SEED = 1
np.random.seed(SEED)
random.seed(SEED)


# ----------------------------------
# Parameter
# ----------------------------------

B=100                   # Number of bootsrap samples to draw
Lmbd_min=10**(-8)       # smallest regularization parameter lambda to be considered
Lmbd_max=10**(1)        # largest regularization paramter lambda to be considered
n_lmbd=50               # number of lambda to test
L=30                    # number of coefficient for the reuglarized torrent
                                                                           
Lmbd=np.array([np.exp(i/n_lmbd*(np.log(Lmbd_max)-np.log(Lmbd_min))+np.log(Lmbd_min)) for i in range(0, n_lmbd)])        # grid of regularization paramters   
noise_vars = 4          # Variance of the noise
n = 2**6                # number of observations n

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
# run experiments
# ----------------------------------

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

# Associate memory
err_cap, err_inl, err_m=np.full([n_lmbd, B], float(0)), np.full([n_lmbd, B], float(0)) ,np.full([n_lmbd, B], float(0))
estimates_decor = get_results( **data_values, **method_args, L=L, lmbd=0, K=K)

# Draw B boostrap samples
for i in range(n_lmbd):
    boot=err_boot(transformed=estimates_decor['transformed'], a=method_args['a'], lmbd=Lmbd[i], K=K, B=B)
    err_cap[i,:], err_inl[i,:], err_m[i,:]=boot["err_cap"], boot["err_inl"], boot["err_m"]

# Compute the average estimated error
err_cap_m, err_inl_m, err_m_m=np.mean(err_cap, axis=1), np.mean(err_inl, axis=1), np.mean(err_m, axis=1)

# Compute the standard deviation
err_cap_sd, err_inl_sd, err_m_sd=np.linalg.norm(err_cap-np.repeat(err_cap_m.reshape(-1,1), B, axis=1), axis=1, ord=2)/(np.sqrt(B-1)), np.linalg.norm(err_inl-np.repeat(err_inl_m.reshape(-1,1), B, axis=1), axis=1, ord=2)/(np.sqrt(B-1)), np.linalg.norm(err_m-np.repeat(err_m_m.reshape(-1,1), B, axis=1), axis=1, ord=2)/(np.sqrt(B-1))

# Compute the indices minimizing the estimated perdiction error
indx_cap, indx_inl, indx_m=np.argmin(err_cap_m), np.argmin(err_inl_m), np.argmin(err_m_m)
lmbd_cap, lmbd_inl, lmbd_m=Lmbd[indx_cap], Lmbd[indx_inl], Lmbd[indx_m]
lmbd_cap_1sd, lmbd_inl_1sd, lmbd_m_1sd=Lmbd[min(np.arange(indx_cap,n_lmbd)[err_cap_m[indx_cap:n_lmbd]>err_cap_m[indx_cap]+err_cap_sd[indx_cap]])], Lmbd[min(np.arange(indx_inl,n_lmbd)[err_inl_m[indx_inl:n_lmbd]>err_inl_m[indx_inl]+err_inl_sd[indx_inl]])], Lmbd[min(np.arange(indx_m,n_lmbd)[err_m_m[indx_m:n_lmbd]>err_m_m[indx_m]+err_m_sd[indx_m]])]

# ----------------------------------
# plotting
# ----------------------------------

fig, axs = plt.subplots(1, 3, figsize=(12, 5))

#P lot the clipping
axs[0].plot(Lmbd, err_cap_m)
axs[0].fill_between(Lmbd, y1=err_cap_m-err_cap_sd, y2=err_cap_m+err_cap_sd, color=ibm_cb[1], alpha=0.1)
axs[0].axvline(x=lmbd_cap, linestyle="dashed", color=ibm_cb[2])
axs[0].text(lmbd_cap/4, 0.5*min(err_cap_m)+0.5*max(err_cap_m), "$\lambda_{min}$", color=ibm_cb[2], rotation=90)
axs[0].axvline(x=lmbd_cap_1sd, linestyle="dashed", color=ibm_cb[3])
axs[0].text(lmbd_cap_1sd/4, 0.5*min(err_cap_m)+0.5*max(err_cap_m), "$\lambda_{1sd}$", color=ibm_cb[3], rotation=90)
axs[0].set_xscale('log')

# Plot the inliers
axs[1].plot(Lmbd, err_inl_m)
axs[1].fill_between(Lmbd, y1=err_inl_m-err_inl_sd, y2=err_inl_m+err_inl_sd, color=ibm_cb[1], alpha=0.1)
axs[1].axvline(x=lmbd_inl, linestyle="dashed", color=ibm_cb[2])
axs[1].text(lmbd_inl/4, 0.5*min(err_inl_m)+0.5*max(err_inl_m), "$\lambda_{min}$", color=ibm_cb[2], rotation=90)
axs[1].axvline(x=lmbd_inl_1sd, linestyle="dashed", color=ibm_cb[3])
axs[1].text(lmbd_inl_1sd/4, 0.5*min(err_inl_m)+0.5*max(err_inl_m), "$\lambda_{1sd}$", color=ibm_cb[3], rotation=90)
axs[1].set_xscale('log')

# Plot the median
axs[2].plot(Lmbd, err_m_m)
axs[2].fill_between(Lmbd, y1=err_m_m-err_m_sd, y2=err_m_m+err_m_sd, color=ibm_cb[1], alpha=0.1)
axs[2].axvline(x=lmbd_m, linestyle="dashed", color=ibm_cb[2])
axs[2].text(lmbd_m/4, 0.5*min(err_m_m)+0.5*max(err_m_m), "$\lambda_{min}$", color=ibm_cb[2], rotation=90)
axs[2].axvline(x=lmbd_m_1sd, linestyle="dashed", color=ibm_cb[3])
axs[2].text(lmbd_m_1sd/4, 0.5*min(err_m_m)+0.5*max(err_m_m), "$\lambda_{1sd}$", color=ibm_cb[3], rotation=90)
axs[2].set_xscale('log')

#Labeling
axs[0].set_xlabel("$\lambda$")
axs[1].set_xlabel("$\lambda$")
axs[2].set_xlabel("$\lambda$")
axs[0].set_ylabel("Estimated Generalization Error")
axs[0].set_title("Clipping")
axs[1].set_title("Omitting")
axs[2].set_title("Median")
plt.tight_layout()
plt.show()