import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy as sp


from utils_nonlinear import get_results, plot_results, get_data, plot_settings, estimated_pred_err
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
    "fraction": 0.1,
    "beta": np.array([2]),
    "band": list(range(0, 50))  # list(range(0, 50)) | None
}

method_args = {
    "a": 0.85,
    "method": "torrent_reg",        # "torrent" | "bfs"
}

lmbd=np.concatenate((np.array([0]), np.array([10**(i/20) for i in range(-250,  0)])))
n_lmbd=len(lmbd)
noise_vars =  1
n = 2 ** 9# number of observations
print("number of observations:", n)

# ----------------------------------
# run experiments
# ----------------------------------
n_x=200
test_points=np.array([i / n_x for i in range(0, n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])
L_temp=50
print("number of coefficients:", L_temp)
#Compute the basis
basis_tmp = [np.cos(np.pi * test_points * k ) for k in range( L_temp)] 
basis = np.vstack(basis_tmp).T
#Get data
data_values = get_data(n, **data_args, noise_var=noise_vars)
data_values.pop('u')
#Set up the smothness penalty
diag=np.concatenate((np.array([0]), np.array([i**4 for i in range(1,L_temp)])))
K=np.diag(diag)
#Allocate memory
estimates_decor=[]
S=set(np.arange(0,n))
err_S=np.full(n_lmbd, np.nan)
err_individual=np.full(n_lmbd, np.nan)
err_stable=np.full(n_lmbd, np.nan)

#Compute the estimates for different lambda's
for i in range(0,n_lmbd):
    estimates_decor.append(get_results(**data_values, **method_args, K=K, L=L_temp, lmbd=lmbd[i]))
    S_i=estimates_decor[i]["inliniers"]
    S=S.intersection(S_i)

print("|S|=" + str(len(S)))
y_transformed=estimates_decor[i]["transformed"]["yn"]
x_transformed=estimates_decor[i]["transformed"]["xn"]

#Compute the estimated prediction error
for i in range(0, n_lmbd):
    coef=estimates_decor[i]["estimate"]
    err_S[i]=1/len(S)*np.linalg.norm(y_transformed[list(S)]-x_transformed[list(S), ]@coef, ord=2)**2
    S_i=estimates_decor[i]["inliniers"]
    err_individual[i]=estimated_pred_err(x_transformed[list(S_i)], y_transformed[list(S_i),], lmbd=lmbd[i], K=K, k=10)

 
k=10
n_S=len(S)
partition_S=np.random.permutation(n_S)
test_fold_size=n_S//k
for i in range(0, n_lmbd):
    S_i={inlinier for inlinier in estimates_decor[i]["inliniers"]}
    S_i_C=S_i.difference(S)
    n_train=len(S_i_C)
    train_fold_size=n_train//k
    partition_S_C=np.random.permutation(n_train)
    err=0
    for j in range(0,k):
        test_indx=[list(S)[i] for i in partition_S[j*test_fold_size:(j+1)*test_fold_size]]
        test_indx_S=[list(S_i_C)[i] for i in partition_S_C[j*train_fold_size:(j+1)*train_fold_size]]
        train_indx=np.concatenate((np.delete(list(S), partition_S[j*test_fold_size:(j+1)*test_fold_size]), np.delete(list(S_i_C), partition_S_C[j*train_fold_size:(j+1)*train_fold_size])))
        X_train=x_transformed[train_indx]
        Y_train=y_transformed[train_indx]
        B=X_train.T @ Y_train
        A=X_train.T @ X_train + lmbd[i]*K 
        coef=sp.linalg.solve(A, B)
        err_add = np.linalg.norm(y_transformed[test_indx] - x_transformed[test_indx] @ coef, ord=2)**2
        err=err+1/n_S*err_add
    err_stable[i]=err

# ----------------------------------
# plotting
# ----------------------------------

fig, ax1 = plt.subplots()

ax1.set_xlabel('$\log(\lambda)$')
ax1.set_ylabel('stable cv', color=ibm_cb[1])
ax1.plot(np.log(lmbd), err_stable, color=ibm_cb[1])
ax1.tick_params(axis='y', labelcolor=ibm_cb[1])

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

ax2.set_ylabel('intersection', color=ibm_cb[4])  # we already handled the x-label with ax1
ax2.plot(np.log(lmbd), err_S, color=ibm_cb[4])
ax2.tick_params(axis='y',  labelcolor=ibm_cb[4])

fig.suptitle("Estimated Prediction Error")
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
