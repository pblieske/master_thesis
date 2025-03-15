import os, random, pickle, json
import numpy as np
from tqdm import tqdm
from pygam import LinearGAM, s
from utils_nonlinear import get_results, get_data
from synthetic_data import functions_nonlinear
from robust_deconfounding.utils import get_funcbasis

"""
We run a Monte Carlo experiment to investigate the consistency of the nonlinear extensions of DecoR.
The experiments can take up to an hour, therefore the results are saved in the folder "results".
The configurations for the different experiments discussed in the paper are saved in the config.json file and can be chosen through the "exp" variable.
New experiments can be run by simply extending the config.json file.
Plots can afterwards be generated by running the plot_nonlinear.py file with the same "exp" variable.

Short explanation of the different experiments:
"uniform": X_t is i.i.d. uniformly distributed on [0,1] and f(x)=6 sin(2\pi x)
"reflected_ou": X_t is a reflected Ornstein-Uhlenbeck process and f(x)=6 sin(2\pi x)
"sigmoid": X_t as in "uniform" and f(x) a sigmoid function
"poly": X_t is a reflected Ornstein-Uhlenbeck process, f(x) a sigmoid function and the monomial basis for the approximation of the function

More details can be found in the config.json file, here a short explanation of the variables defined there and what they do:

Data generation:
The "process_type" can either be an Ornstein-Uhlenbeck process or a band-limited process.
The "basis_type" is the basis for which the confounder is sparse. This is also the basis used by DecoR.
The "fraction" variable is the fraction of outliers. For example "0.25" means that a fourth of the datapoints 
is confounded in the "basis"-domain.
The "beta" variable chooses the underlying true, nonlinear function, see the functions_nonlinear in the synthetic_data.py file.
The "noise_var" is the variance of the noise, i.e. $\sigma_{\eta}^2$.
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


exp="test"     # "uniform" | "reflected_ou" | "sigmoid" | "poly"


# ----------------------------------
# Set up and prepare Monte Carlo simulation
# ----------------------------------

path= os.path.dirname(__file__)                    #Path for the json file where experiment configurations are defined.
path_results=os.path.join(path, "results/")        #Path to the results

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

#Read in the parameters from the config.json file
with open(path+'/config.json', 'r') as file:
    config = json.load(file)

config=config["experiment_"+str(exp)]
data_args, method_args, m, noise_vars, L_frac, num_data= config["data_args"], config["method_args"], config["m"], np.array(config["noise_vars"]), np.array(config["L_frac"]), np.array(config["num_data"])   
data_args["beta"]=np.array(data_args["beta"])                       


# ----------------------------------
# run experiments
# ----------------------------------

n_x=200                 # Resolution of x-axis
int_test=[0, 1]         # Interval [a,b] on which on compute the L^1-error
len_test=int_test[1]-int_test[0]
test_points=np.array([int_test[0]+ i/(n_x)*len_test for i in range(n_x)])
y_true=functions_nonlinear(np.ndarray((n_x,1), buffer=test_points), data_args["beta"][0])

for i in range(len(noise_vars)):
    print("Noise Variance: ", noise_vars[i])
    res = {"DecoR": [], "ols": []}

    for n in num_data:
        res["DecoR"].append([])
        res["ols"].append([])
        L=max((np.ceil(n**0.5)/L_frac[i]).astype(int),2)
        basis=get_funcbasis(x=test_points, L=L, type=method_args["basis_type"])

        print("number of data points: ", n)
        print("number of coefficients: ", L)

        for _ in tqdm(range(m)):

            # Get the data
            data_values = get_data(n, **data_args, noise_var=noise_vars[i])
            data_values.pop('u') 
            outlier_points=data_values.pop('outlier_points')

            # Fit DecoR
            estimates_decor = get_results(**data_values, **method_args, L=L)
            y_est=basis @ estimates_decor["estimate"]
            y_est=np.ndarray((n_x, 1), buffer=y_est)

            # Fit the benchmark model
            x=np.reshape(data_values["x"], (-1,1))
            y=data_values["y"]
            gam = LinearGAM(s(0)).gridsearch(x, y, progress=False)
            y_bench=gam.predict(test_points)
           
            res["DecoR"][-1].append(1/n_x*len_test*np.linalg.norm(y_true-y_est, ord=1))
            res["ols"][-1].append(1/n_x*len_test*np.linalg.norm(y_true-y_bench, ord=1))

    #Save the results using a pickle file
    res["DecoR"], res["ols"] = np.array(res["DecoR"]), np.array(res["ols"])
    with open(path_results+"experiment_" + exp +'_noise_='+str(noise_vars[i])+'.pkl', 'wb') as fp:
        pickle.dump(res, fp)
        print('Results saved successfully to file.')
