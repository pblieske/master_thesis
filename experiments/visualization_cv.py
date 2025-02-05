import numpy as np
import random
import matplotlib.pyplot as plt

from robust_deconfounding.utils import cosine_basis, get_funcbasis
from utils_nonlinear import get_data, plot_settings, conf_help, get_results
from synthetic_data import functions_nonlinear
from robust_deconfounding.robust_regression import Torrent_reg
from robust_deconfounding.decor import DecoR


"""
    A visual example of cross-validation on the transformed data.
"""

SEED = 9
np.random.seed(SEED)
random.seed(SEED)


# ----------------------------------
# Parameter
# ----------------------------------

Lmbd_min=10**(-8)       # smallest regularization parameter lambda to be considered
Lmbd_max=10**(1)        # largest regularization paramter lambda to be considered
n_lmbd=200              # number of lambda to test
L=30                    # number of coefficient for the reuglarized torrent
                                                                           
Lmbd=np.array([np.exp(i/n_lmbd*(np.log(Lmbd_max)-np.log(Lmbd_min))+np.log(Lmbd_min)) for i in range(0, n_lmbd)])        # grid of regularization paramters   
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

#Perform the Cross-Validation
robust_algo = Torrent_reg(a=method_args["a"], fit_intercept=False, K=K, lmbd=0)
cv=robust_algo.cv(x=R, y=data_values["y"], Lmbd=Lmbd)
err_cv=cv["pred_err"]
indx_min=np.argmin(err_cv)
lmbd_cv=Lmbd[indx_min]

#Estimate the variance
sigma=np.zeros((n_lmbd,1))
for i in range(n_lmbd): 
    estimates_decor = get_results( **data_values, **method_args, L=L, lmbd=Lmbd[i], K=K)
    conf=conf_help(**estimates_decor, L=L, alpha=0.95, K=K)
    sigma[i]=conf["sigma"]
sigma=sigma[:,0]/2

try:
    indx_se=max(np.array(range(indx_min))[np.array(err_cv[0:indx_min]>=err_cv[indx_min]+sigma[indx_min])])
except:
    indx_se=0

lmbd_se=Lmbd[indx_se]


# ----------------------------------
# plotting
# ----------------------------------

plt.plot(Lmbd, err_cv, color=ibm_cb[1])
plt.fill_between(Lmbd, y1=err_cv-sigma, y2=err_cv+sigma, color=ibm_cb[1], alpha=0.1)
plt.axvline(x=lmbd_cv, linestyle="dashed", color=ibm_cb[2])
plt.axvline(x=lmbd_se, linestyle="dashed", color=ibm_cb[4])
 
# Labels
plt.text(lmbd_cv/4, 0.5*min(err_cv)+0.5*max(err_cv), "$\lambda_{min}$", color=ibm_cb[2], rotation=90)
plt.text(lmbd_se/4, 0.5*min(err_cv)+0.5*max(err_cv), "$\lambda_{se}$", color=ibm_cb[4], rotation=90)
plt.xscale("log")
plt.xlabel("$\lambda$")
plt.ylabel("Estimated Prediction Error")
plt.title("Noise Variance " + str(noise_vars))
plt.tight_layout()
plt.show()