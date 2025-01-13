import sys
sys.path.insert(0, '/mnt/c/Users/piobl/Documents/msc_applied_mathematics/4_semester/master_thesis/code/master_thesis')

from robust_deconfounding import DecoR
from robust_deconfounding.robust_regression import Torrent
from robust_deconfounding.utils import cosine_basis, get_funcbasis, get_funcbasis_multivariate
from utils_nonlinear import get_results, plot_settings, get_conf


from pygam import GAM, s, intercept, te
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

#Set labels, adjust grid and title
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

#Adjust for delay from x to y in days
delay=1
x=x[0:(n-delay)]
y=y[delay:n]
x_min=np.min(x)
x_max=np.max(x)
x_norm=(x-x_min)/(x_max-x_min)

temp=u[delay:n]
t_min=np.min(temp)
t_max=np.max(temp)
temp_norm=(temp-t_min)/(t_max-t_min)

#Compute the design matrix
X=np.stack((x_norm, temp_norm))

# ----------------------------------
# Deconfounding and Estimation of Causal Relationship
# ----------------------------------

method_args = {
    "a": 0.95,
    "method": "torrent",        # "torrent" | "bfs"
    "basis_type": "cosine_cont",# basis used for the approximation of f
}

bench_mark="spline"         #Benchmark type
L=6                        #Number of coefficinet for DecoR regression only on ozone levels
L_adjst=np.array([6, 8])    #Number of coefficients, [ozone, temperature]

#if method_args["method"] in ["torrent_cv", "torrent_reg"]:
diag=np.concatenate((np.array([0]), np.array([i**2 for i in range(1,L_adjst[0]+1)]), np.array([i**4 for i in range(1,L_adjst[1]+1)])))
K=np.diag(diag)
Lmbd=np.array([10**(i/40) for i in range(-300, 80)])
n_x=200     #Resolution of x-axis

#Compute matrices to obtain estimations of y
test_points=np.linspace(0, 1, num=n_x)
test_points_adjst=np.stack((test_points, np.repeat(np.mean(temp_norm), n_x)))
test_points_adjst_temp=np.stack((np.repeat(np.mean(x_norm), n_x), test_points))
basis=get_funcbasis(x=test_points, L=L, type=method_args["basis_type"])
basis_adjst=get_funcbasis_multivariate(x=test_points_adjst, L=L_adjst, type=method_args["basis_type"])
basis_temp=get_funcbasis_multivariate(x=test_points_adjst_temp, L=L_adjst, type=method_args["basis_type"])

#Fit DecoR without adjustement for the temperature
estimates_decor=get_results(x=x_norm, y=y, **method_args, basis=cosine_basis(n-1), L=L, K=K, lmbd=0.001)
y_est=basis @ estimates_decor["estimate"]
ci=get_conf(x=test_points, **estimates_decor, alpha=0.95, basis_type=method_args["basis_type"])

#Fit DecoR with adjustement for the temperature
estimates_decor_adjst=get_results(x=X, y=y, **method_args, basis=cosine_basis(n-1), L=L_adjst, K=K, lmbd=10**(-4))
y_adjst=basis_adjst @ estimates_decor_adjst["estimate"]
ci_adjst=get_conf(x=test_points_adjst, **estimates_decor_adjst, alpha=0.9, L=L_adjst, basis_type=method_args["basis_type"])

#Fit benchmark for comparison and to make the confounding visable    
if bench_mark=="spline":
    gam = GAM(s(0)+intercept, lam=5).fit(np.reshape(X.T, (-1,2)), y) #, lam=Lmbd) 
    y_bench=gam.predict(test_points_adjst.T)
    ci_bench=gam.confidence_intervals(test_points_adjst.T, width=0.95)
else:
    estimates_fourier= get_results(x=x_norm, y=y, basis=cosine_basis(n-2), method="ols", L=L, basis_type=method_args["basis_type"], a=method_args["a"])
    y_bench=basis @ estimates_fourier["estimate"]
    ci_bench=get_conf(x=test_points, **estimates_fourier, alpha=0.95, basis_type=method_args["basis_type"])


# ----------------------------------
# A Try to estimate the number of death caused by ozone
# ----------------------------------

ind=(x>=60)     #index of observation with ozone lever higher than 60 \mu g/m^3
x_high=(x[ind]-x_min)/(x_max-x_min)
t_high=temp_norm[ind]
x_lim=(60-x_min)/(x_max-x_min)
n_high=len(x_high)

#Estimate the effect of lowering the ozon level to 60
basis_pred=get_funcbasis_multivariate(x=np.stack((x_high, t_high)), L=L_adjst, type=method_args["basis_type"])
basis_limit=get_funcbasis_multivariate(x=np.stack((np.repeat(x_lim, n_high), t_high)), L=L_adjst, type=method_args["basis_type"])
y_caused=basis_pred @ estimates_decor_adjst["estimate"] -basis_limit @ estimates_decor_adjst["estimate"]

#Compute the upper and lower bound using a 0.95 confidence interval
ci_pred=get_conf(x=np.stack((x_high, t_high)), **estimates_decor_adjst, L=L_adjst, basis_type=method_args["basis_type"])
ci_lim=get_conf(x=np.stack((np.repeat(x_lim, n_high), t_high)), **estimates_decor_adjst, L=L_adjst, basis_type=method_args["basis_type"])

#Compute the mean per year
num_deaths={"mean": sum(y_caused)/5, "lower bound": sum(ci_pred[:,0] -ci_lim[:,1])/5, "upper bound": sum(ci_pred[:,1] -ci_lim[:,0])/5}
print("Number of deaths caused per year trough a ozone levels over 60 mg/m^2: " , num_deaths)

# ----------------------------------
# plotting
# ----------------------------------

test_ozone=(test_points)*(x_max-x_min)+x_min

plt.scatter(x=x, y=y, color='w', edgecolors="gray", s=4) 
plt.plot(test_ozone, y_bench, '-', color=ibm_cb[4], linewidth=1.5)
plt.plot(test_ozone, y_adjst, '-', color=ibm_cb[1], linewidth=1.5)

#Plot confidence intervals
plt.fill_between(test_ozone, y1=ci_bench[:, 0], y2=ci_bench[:, 1], color=ibm_cb[4], alpha=0.4)
plt.fill_between(test_ozone, y1=ci_adjst[:, 0], y2=ci_adjst[:, 1], color=ibm_cb[1], alpha=0.4)

def get_handles():
    point_1 = Line2D([0], [0], label='Observations', marker='o', mec="gray", markersize=3, linestyle='')
    point_3 = Line2D([0], [0], label="DecoR" , color=ibm_cb[1], linestyle='-')
    point_4= Line2D([0], [0], label="GAM" , color=ibm_cb[4], linestyle='-')
    return [point_1,  point_3, point_4]

plt.xlabel("Ozone ($\mu g/m^3$)")
plt.ylabel("# Deaths")
plt.title("Influence of Ozone on Health")
plt.legend(handles=get_handles(), loc="upper left")
plt.grid(linestyle='dotted')
plt.tight_layout()
plt.show()

#Plot the selected outliers
inl=estimates_decor_adjst["inliers"]
out=np.delete(np.arange(0,n), list(inl))
plt.hist(out,  color=ibm_cb[0], edgecolor='k', alpha=0.6, bins=15)
plt.xlabel("Frequency")
plt.ylabel("Count")
plt.title("Histogramm of Excluded Frequencies")
plt.tight_layout()
plt.show()

#Plot the influence of temperature on #deaths
y_temp=basis_temp @ estimates_decor_adjst["estimate"]
test_temp=(test_points)*(t_max-t_min)+t_min
ci_temp=get_conf(x=test_points_adjst_temp, **estimates_decor_adjst, alpha=0.95, L=L_adjst, basis_type=method_args["basis_type"])

plt.scatter(x=temp, y=y, marker='o', color='w', edgecolors="gray", s=5) 
plt.plot(test_temp, y_temp, '-', color=ibm_cb[2], linewidth=1.5)
plt.fill_between(test_temp, y1=ci_temp[:, 0], y2=ci_temp[:, 1], color=ibm_cb[2], alpha=0.3)

plt.grid(linestyle='dotted')
plt.ylabel("# Deaths")
plt.xlabel("temperature ($C^\circ$)")
plt.title("Influence of Temperature")
plt.tight_layout()
plt.show()


#Comparing the with and without adjustement
y_diff=y_est-y_adjst
plt.plot(test_ozone, y_diff, '-', color=ibm_cb[1], linewidth=1.5)
plt.fill_between(test_ozone, y1=ci[:, 0]-ci_adjst[:, 0], y2=ci[:, 1]-ci_adjst[:, 1], color=ibm_cb[1], alpha=0.3)

def get_handles():
    point_1 = Line2D([0], [0], label='Observations', marker='o', mec="gray", markersize=3, linestyle='')
    point_3 = Line2D([0], [0], label="$y-y^{adjst}$" , color=ibm_cb[1], linestyle='-')
    point_4= Line2D([0], [0], label="adjst." , color=ibm_cb[2], linestyle='-')
    return [point_1,  point_3, point_4]

plt.xlabel("Ozone ($\mu g/m^3$)")
plt.ylabel("Difference")
plt.title("DecoR with/-out Adjustement")
plt.grid(linestyle='dotted')
plt.tight_layout()
plt.show()


#Plot the derivative
"""
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
"""