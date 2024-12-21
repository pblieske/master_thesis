import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils_nonlinear import  plot_results,  plot_settings

"""
We plot the results, i.e. L^2-error, obtained from cosistency.py
Attention: File has to be run with the same parameters as consistency.py to esnure the correct files are read and are consistent with the plot settings.
"""

path="/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/results/"   #Path to save files
colors, ibm_cb = plot_settings()

m = 2   #Number of repetitions for the Monte Carlo
noise_vars = [0, 1, 4]
num_data = [2 ** k for k in range(5, 10)]      # up to k=14 

# ----------------------------------
# Load data and plotting
# ----------------------------------

for i in range(len(noise_vars)):
    with open(path+'noise='+str(noise_vars[i])+'.pkl', 'rb') as fp:
        res = pickle.load(fp)
    plot_results(res, num_data, m, colors=colors[i])

# ----------------------------------
# plotting
# ----------------------------------

def get_handles():
    point_1 = Line2D([0], [0], label='OLS', marker='o',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_2 = Line2D([0], [0], label='DecoR', marker='X',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_3 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[0]), markersize=10,
                     color=ibm_cb[1], linestyle='-')
    point_4 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[1]), markersize=10,
                     color=ibm_cb[4], linestyle='-')
    point_5 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[2]), markersize=10,
                     color=ibm_cb[2], linestyle='-')
    return [point_1, point_2, point_3, point_4, point_5]


plt.xlabel("number of data points")
plt.ylabel("L^2 error")
plt.title("$L^2$-consistency")
plt.xscale('log')
plt.xlim(left=num_data[0] - 2)
plt.hlines(0, num_data[0], num_data[-1], colors='black', linestyles='dashed')
plt.legend(handles=get_handles(), loc="lower left")
plt.tight_layout()

plt.show()