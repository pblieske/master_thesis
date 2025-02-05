import numpy as np
import os, random, pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils_nonlinear import get_results, get_data, plot_settings

"""
    This is a supplementary experiment to the cross validation study inverstigating the size of the the stable set S of iliers.
"""

run_exp=True

Lmbd_min=10**(-8)       # smallest regularization parameter lambda to be considered
Lmbd_max=10**(1)        # largest regularization paramter lambda to be considered
n_lmbd=100              # number of lambda to test
L_cv=30                 # number of coefficient for the reuglarized torrent
m=200                   # number of Monte Carlo samples to draw
                                                                           
Lmbd=np.concatenate((np.array([0]),np.array([np.exp(i/n_lmbd*(np.log(Lmbd_max)-np.log(Lmbd_min))+np.log(Lmbd_min)) for i in range(0, n_lmbd)])))        # grid of regularization paramters   
noise_var = 1                       # Variance of the noise
num_data = [64, 256, 1024, 8192]    # number of observations n

data_args = {
    "process_type": "oure",         # "uniform" | "oure"
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

colors, ibm_cb = plot_settings()                                    # import colors for plotting
path_results=os.path.join(os.path.dirname(__file__), "results/")    # path to the results

SEED = 1
np.random.seed(SEED)
random.seed(SEED)


# ----------------------------------
# run experiments
# ----------------------------------

n_x=200             # Resolution of x-axis
test_points = np.array([i / n_x for i in range(0, n_x)])
size=np.zeros(shape = [len(num_data), m ]) 

if run_exp:
    for i in range(0, len(num_data)):
        n=num_data[i]
        print("number of data points: ", n)

        #Construct smothness penalty
        diag=np.concatenate((np.array([0]), np.array([i**4 for i in range(1,L_cv+1)])))
        K=np.diag(diag)

        #Run Monte Carlo simulation
        for k in tqdm(range(m)):
            data_values = get_data(n, **data_args, noise_var=noise_var)
            data_values.pop('u', 'basis')
            S=set(np.arange(0,n))  
            for j in range(n_lmbd):
                estimates_tor = get_results(**data_values, **method_args, L=L_cv, lmbd=Lmbd[j], K=K)
                S=S.intersection(estimates_tor["inliers"])            
            size[i, k]=len(S)

    # Saving the results to a pickle file
    with open(path_results+"experiment_size_S_="+'.pkl', 'wb') as fp:
        pickle.dump(S, fp)
        print('Results saved successfully to file.')

else:
    # Loading the file with the saved results
    with open(path_results+"experiment_size_S_="+'.pkl', 'rb') as fp:
        S = pickle.load(fp)


# ----------------------------------
# plotting
# ----------------------------------

fig, axs = plt.subplots(2, 2)

axs[0, 0].hist(size[0,:],  color=ibm_cb[0], edgecolor='k', alpha=0.6)
axs[0, 0].set_title(str(num_data[0])+ " Observations")
axs[0, 1].hist(size[1,:],  color=ibm_cb[0], edgecolor='k', alpha=0.6)
axs[0, 1].set_title(str(num_data[1])+ " Observations")
axs[1, 0].hist(size[2,:],  color=ibm_cb[0], edgecolor='k', alpha=0.6)
axs[1, 0].set_title(str(num_data[2])+ " Observations")
axs[1, 1].hist(size[3,:],  color=ibm_cb[0], edgecolor='k', alpha=0.6)
axs[1, 1].set_title(str(num_data[3])+ " Observations")

#Labels
for i in range(0,2):
    for j in range(0,2):
        axs[i, j].set_xlabel('$|S|$')
        axs[i, j].set_ylabel('count')

plt.suptitle("Size of the stable inliers set S")
plt.tight_layout()
plt.show()