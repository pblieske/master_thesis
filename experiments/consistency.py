import numpy as np
import random, pickle, json
from pygam import LinearGAM, s

import sys
sys.path.insert(0, '/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/code/master_thesis')
from utils_nonlinear import get_results, get_data, get_parameters
from synthetic_data import functions_nonlinear
from robust_deconfounding.utils import get_funcbasis

"""
Short explanation of the variables and what they do:

Data generation:
The "process_type" can either be an Ornstein-Uhlenbeck process or a band-limited process.
The "basis_type" is the basis for which the confounder is sparse. This is also the basis used by DecoR.
The "fraction" variable is the fraction of outliers. For example "0.25" means that a fourth of the datapoints 
is confounded in the "basis"-domain.
The "beta" variable is the $\beta$ value i.e. the true causal effect. It can be two- or one-dimensional.
The "noise_var" is the variance of the noise i.e. $\sigma_{\eta}^2$.
The "band" variable is the indices of the band for the band-limited process. Does nothing if "ou" is selected for
the "process_type".

Algorithm:
The "a" variable is the upper bound for the fraction of inliers in the data.
The "method" variable can either be "torrent" or "bfs" the two robust-regression algorithms implemented. DecoR 
can be easily extended to include other robust regression techniques.

Experiments:
The "m" is the number of times we resample the data to get confidence intervals.
The "num_data" variable is a list of increasing natural numbers that indicate the amount of data tested on.
"""

path="/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/results/"   #Path to save files
path_config="/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/code/master_thesis/experiments/" #Path for the json file
exp="1"     #Select the experiment

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

#Read in the parameters from the config.json file
with open(path_config+'config.json', 'r') as file:
    config = json.load(file)

config=config["experiment_"+str(exp)]
data_args, method_args, m, noise_vars, L_frac, num_data= config["data_args"], config["method_args"], config["m"], np.array(config["noise_vars"]), np.array(config["L_frac"]), np.array(config["num_data"])   
data_args["beta"]=np.array(data_args["beta"])                       


# ----------------------------------
# run experiments
# ----------------------------------

n_x=200     #Resolution of x-axis
int_test=[0.1, 0.9]
len_test=int_test[1]-int_test[0]
test_points=np.array([int_test[0]+ i/(n_x)*len_test for i in range(n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])

for i in range(len(noise_vars)):
    print("Noise Variance: ", noise_vars[i])
    res = {"DecoR": [], "ols": []}       

    for n in num_data:
        print("number of data points: ", n)
        res["DecoR"].append([])
        res["ols"].append([])
        L_temp=max((np.ceil(n**0.5)/L_frac[i]).astype(int),4) #max((np.floor(np.log(n))).astype(int),1) max((np.floor(n**(1/2)/4)).astype(int),1)
        basis=get_funcbasis(x=test_points, L=L_temp, type=method_args["basis_type"])
        #n_con=min((2*n**(-0.5)), 1)
        print("number of coefficients: ", L_temp)
        #print("number of confounded frequencies: ", n_con)

        for _ in range(m):
            data_values = get_data(n, **data_args, noise_var=noise_vars[i]) #, fraction=n_con)
            data_values.pop('u') 
            outlier_points=data_values.pop('outlier_points')
            estimates_decor = get_results(**data_values, **method_args, L=L_temp) # a=1-n_con*1.25)
            y_est=basis @ estimates_decor["estimate"]
            y_est=np.ndarray((n_x, 1), buffer=y_est)
            """ 
            estimates_fourrier= get_results(**data_values, method="ols", L=L_temp, a=0, outlier_points=outlier_points, basis_type=method_args["basis_type"])
            y_bench= basis @ estimates_fourrier["estimate"]
            y_bench=np.ndarray((n_x, 1), buffer=y_bench)
            """
            x=np.reshape(data_values["x"], (-1,1))
            y=data_values["y"]
            gam = LinearGAM(s(0)).gridsearch(x, y)
            y_bench=gam.predict(test_points)
           
            """
            res["DecoR"][-1].append(1/np.sqrt(n_x)*np.linalg.norm(y_true-y_est, ord=2))
            res["ols"][-1].append(1/np.sqrt(n_x)*np.linalg.norm(y_true-y_bench, ord=2))
            """
            res["DecoR"][-1].append(1/n_x*len_test*np.linalg.norm(y_true-y_est, ord=1))
            res["ols"][-1].append(1/n_x*len_test*np.linalg.norm(y_true-y_bench, ord=1))

    #Save the results using a pickle file
    res["DecoR"], res["ols"] = np.array(res["DecoR"]), np.array(res["ols"])
    with open(path+"experiment_" + exp +'noise='+str(noise_vars[i])+'.pkl', 'wb') as fp:
        pickle.dump(res, fp)
        print('Results saved successfully to file.')
