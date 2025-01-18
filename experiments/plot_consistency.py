import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

from utils_nonlinear import  plot_results,  plot_settings, plot_results_2yaxis

"""
We plot the results, i.e. L^2-error, obtained from cosistency.py
Attention: File has to be run with the same parameters as consistency.py to esnure the correct files are read and are consistent with the plot settings.
"""

path="/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/results/"   #Path to load files from
colors, ibm_cb = plot_settings()

#Parameters used to run the experiments
m = 200                                      #Number of repetitions for the Monte Carlo
noise_vars = [0, 1, 4]                      
num_data= num_data = [2 ** k for k in range(4, 9)] + [2**10] +[2**13]  # up to k=14 

# ----------------------------------
# Load data and plotting
# ----------------------------------

#Theoritacl convergence speed from Thm 4.2
#x=np.array([2 ** (k/10) for k in range(60, 131)] ) 
#y=0.2*(x)**(-1/4)*np.log((x)**(1/4))+(x)**(-1/4)
#plt.plot(x, y, color='0.6', linestyle='-')

seperate_axis=False

if seperate_axis==False:
    plt.hlines(0, num_data[0], num_data[-1], colors='black', linestyles='dashed')
    for i in range(len(noise_vars)):
        with open(path+'noise='+str(noise_vars[i])+'.pkl', 'rb') as fp:
            res = pickle.load(fp)
        plot_results(res, num_data, m, colors=colors[i])
else:
    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    ax1.hlines(0, num_data[0], num_data[-1], colors='black', linestyles='dashed')

    for i in range(len(noise_vars)):
        with open(path+'noise='+str(noise_vars[i])+'.pkl', 'rb') as fp:
            res = pickle.load(fp)

        values_decor = np.sqrt(0.8)*np.concatenate([np.expand_dims(res["DecoR"], 2)], axis=2).ravel()
        time = np.repeat(num_data, m)

        df = pd.DataFrame({"value": values_decor.astype(float),
                        "n": time.astype(float)})
        ax1=sns.lineplot(data=df, x="n", y="value", 
                    marker="X", dashes=False, errorbar=("ci", 95), err_style="band",
                    color=colors[i][1],  legend=False, ax=ax1)
        
        values_bench = np.sqrt(0.8)*np.concatenate([np.expand_dims(res["ols"], 2)], axis=2).ravel()

        df = pd.DataFrame({"value": values_bench.astype(float),
                        "n": time.astype(float)})
        ax2=sns.lineplot(data=df, x="n", y="value", 
                    marker="o", dashes=False, errorbar=("ci", 95), err_style="band",
                    color=colors[i][0], legend=False, ax=ax2)   
        
    ax2.set_ylim(0, 4) 


# ----------------------------------
# plotting
# ----------------------------------

def get_handles():
    point_1 = Line2D([0], [0], label='GAM', marker='o',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_2 = Line2D([0], [0], label='DecoR', marker='X',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_6 = Line2D([0], [0], label="Theory" , markersize=10,
                     color='0.6', linestyle='-')
    point_3 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[0]), markersize=10,
                     color=ibm_cb[1], linestyle='-')
    point_4 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[1]), markersize=10,
                     color=ibm_cb[4], linestyle='-')
    point_5 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[2]), markersize=10,
                     color=ibm_cb[2], linestyle='-')
    return [point_1, point_2, point_3, point_4, point_5]

if seperate_axis:
    ax1.set_ylabel("DecoR")
    ax2.set_ylabel("GAM")
    ax1.set_xlabel("number of data points")
else:
    plt.ylabel("$L^1$-error")
    plt.xlabel("number of data points")

plt.title("Non-Parametric ($L=\infty$)") 
plt.xscale('log')
plt.xlim(left=num_data[0] - 2)
plt.legend(handles=get_handles(), loc="center right")
plt.tight_layout()

plt.show()