import sys
sys.path.insert(0, '/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/code/master_thesis')

from robust_deconfounding import DecoR
from robust_deconfounding.robust_regression import Torrent
from robust_deconfounding.utils import cosine_basis
from utils_nonlinear import get_results, plot_settings


import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

"""
    We apply our methods to the ozon dataset.
"""

colors, ibm_cb = plot_settings()

# ----------------------------------
# Read the data
# ----------------------------------

df = pd.read_stata('/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/data_environmental_epidemology/ije-2012-10-0989-File003.dta')
n=df.shape[0]
x=np.array(df.loc[ : , "ozone"])
y=np.array(df.loc[: , "numdeaths"])
u=np.array(df.loc[:, "temperature"])
date=np.array(df.loc[:, "date"])
#print("Corelation temperature and ozone: " + str(np.corrcoef(x,u)))
#print("Corelation temperature and numdeaths: " + str(np.corrcoef(y,u)))


# ----------------------------------
# Plot the two time series
# ----------------------------------

fig, axs = plt.subplots(2, 1)

#Plotting
axs[0].plot(date, x,'o', marker='.', color="black", markersize=3)
axs[1].plot(date, y, 'o', marker='.', color="black", markersize=3)
#Labels, grid and title
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Ozone ($\mu g/m^3$)')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('# Deaths')
axs[0].set_title("Ozone levels over time")
axs[1].set_title("Daily deaths over time")
axs[0].grid(linestyle='dotted')
axs[1].grid(linestyle='dotted')

plt.tight_layout()
plt.show()


# ----------------------------------
# Normalize the data
# ----------------------------------

#Delay from x to y in days
delay=1
x=x[0:(n-(1+delay))]
y=y[delay:(n-1)]
x_min=np.min(x)
x_max=np.max(x)
x=(x-x_min)/(x_max-x_min)


# ----------------------------------
# Deconfounding and Estimation of Causal Relationship
# ----------------------------------

n=len(y)
L=8
Lmbd=np.array([10**(i/40) for i in range(-300, 40)])
diag=np.concatenate((np.array([0]), np.array([i**4 for i in range(1,L)])))
K=np.diag(diag)
result=get_results(x=x, y=y, method="torrent_reg", basis=cosine_basis(n), a=0.95, L=L, K=K, lmbd=0)
estimates_fourrier= get_results(x=x, y=y, basis=cosine_basis(n), method="ridge", L=L, a=0, K=K, lmbd=0)

# ----------------------------------
# plotting
# ----------------------------------

test_points=np.linspace(0, 1, num=200)
#Compute the basis
basis_tmp = [np.cos(np.pi * test_points * k ) for k in range(L)] 
basis = np.vstack(basis_tmp).T
y_est=basis @ result["estimate"]
y_ols=basis @ estimates_fourrier["estimate"]
test_points=(test_points)*(x_max-x_min)+x_min

plt.plot((x*(x_max-x_min)+x_min), y, 'o:w', mec="gray", markersize=3)
plt.plot(test_points, y_est, '-', color=ibm_cb[1], linewidth=2)
plt.plot(test_points, y_ols, '-', color=ibm_cb[4], linewidth=2)

def get_handles():
    point_1 = Line2D([0], [0], label='Observations', marker='o', mec="gray", markersize=3, linestyle='')
    point_3 = Line2D([0], [0], label="DecoR" , color=ibm_cb[1], linestyle='-')
    point_4= Line2D([0], [0], label="OLS" , color=ibm_cb[4], linestyle='-')
    return [point_1,  point_3, point_4]

plt.xlabel("Ozone ($\mu g/m^3$)")
plt.ylabel("# Deaths")
plt.title("Influence of Ozone on Health")
plt.legend(handles=get_handles(), loc="upper left")
plt.grid(linestyle='dotted')
plt.tight_layout()
plt.show()

#Plot the selected outliers
inl=result["inliniers"]
out=np.delete(np.arange(0,n), list(inl))
#sns.histplot(data=pd.DataFrame(data=out))
plt.hist(out,  color=ibm_cb[0], edgecolor='k', alpha=0.6)
plt.xlabel("Frequency")
plt.ylabel("Count")
plt.title("Histogramm of Excluded Frequencies")
plt.show()

#Plot the derivative
test_points=np.linspace(0, 1, num=200)
basis_derivative = [k* -np.sin(np.pi * test_points * k ) for k in range(L)] 
basis_derivative = np.vstack(basis_derivative).T
y_derivative=basis_derivative @ result["estimate"]
y_ols=basis_derivative @ estimates_fourrier["estimate"]
test_points=(test_points)*(x_max-x_min)+x_min

plt.plot(test_points, y_derivative, '-', color=ibm_cb[1])
plt.plot(test_points, y_ols, '-', color=ibm_cb[4])
plt.axhline(y = 0, color = 'black', linestyle = '--')

def get_handles():
    point_1 = Line2D([0], [0], label="DecoRe" , color=ibm_cb[1], linestyle='-')
    point_4= Line2D([0], [0], label="OLS" , color=ibm_cb[4], linestyle='-')
    return [point_1, point_4]

plt.xlabel("Ozone ($\mu g/m^3$)")
plt.ylabel("Derivative")
plt.title("Derivative")
plt.legend(handles=get_handles(), loc="upper left")
plt.tight_layout()
plt.show()
