import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygam import GAM, s, intercept
from matplotlib.lines import Line2D
from tqdm import tqdm

from robust_deconfounding.utils import cosine_basis, get_funcbasis, get_funcbasis_multivariate
from utils_nonlinear import get_results, plot_settings, get_conf, conf_help, err_boot

"""
    We apply the nonlinear extension of DecoR to the ozon dataset to estimate the influence of the daily ozone level (X_t) onto the number of deaths (Y_t).
    For this, we use a delay of the effect of one day and include as a second covariate the daily mean temperature to adjust for heatwaves. The effects
    are asumed to be additive. We compare it with the state of the art estimation from:
        "Time series regression studies in environmental epidemiology"
        Authors: Bhaskaran, Krishnan and Gasparrini, Antonio and Hajat, Shakoor and Smeeth, Liam and Armstrong, Ben
        url: https://academic.oup.com/ije/article/42/4/1187/657875
"""


path_data=os.path.join(os.path.dirname(__file__), "data/")      # Path of data
colors, ibm_cb = plot_settings()                                # Colors for plotting     


# ----------------------------------
# Read the data
# ----------------------------------

df = pd.read_stata(path_data+"ozone.dta")
n=df.shape[0]                               # number of samples
x=np.array(df.loc[ : , "ozone"])            
y=np.array(df.loc[: , "numdeaths"])
u=np.array(df.loc[:, "temperature"])
date=np.array(df.loc[:, "date"])


# ----------------------------------
# Plot the two time series
# ----------------------------------

fig, axs = plt.subplots(2, 1, figsize=(8, 5))

# Plotting
axs[0].plot(date, x,'o', marker='.', color="black", markersize=3)
axs[1].plot(date, y, 'o', marker='.', color="black", markersize=3)

# Set labels, adjust grid and title
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

# Adjust for delay from x to y in days
delay=1     # Delay between ozon exposure and outcome in days
x=x[0:(n-delay)]
y=y[delay:n]
x_min=np.min(x)
x_max=np.max(x)
x_norm=(x-x_min)/(x_max-x_min)

temp=u[delay:n]
t_min=np.min(temp)
t_max=np.max(temp)
temp_norm=(temp-t_min)/(t_max-t_min)

# Compute the design matrix
X=np.stack((x_norm, temp_norm))


# ----------------------------------
# Deconfounding and Estimation of Causal Relationship
# ----------------------------------

method_args = {
    "a": 0.95,
    "method": "torrent",        # "torrent" | "bfs"
    "basis_type": "cosine_cont",# basis used for the approximation of f
}

bench_mark="spline"         # Benchmark type
L=6                         # Number of coefficinet for DecoR regression only on ozone levels
L_adjst=np.array([6, 6])    # Number of coefficients, [ozone, temperature]

diag=np.concatenate((np.array([0]), np.array([i**2 for i in range(1,L_adjst[0]+1)]), np.array([i**4 for i in range(1,L_adjst[1]+1)])))
K=np.diag(diag)
Lmbd=np.array([10**(i/40) for i in range(-300, 80)])
n_x=200                     # Resolution of x-axis

# Compute matrices to obtain estimations of y
test_points=np.linspace(0, 1, num=n_x)
test_points_adjst=np.stack((test_points, np.repeat(0, n_x)))
test_points_adjst_temp=np.stack((np.repeat(0, n_x), test_points))
basis=get_funcbasis(x=test_points, L=L, type=method_args["basis_type"])
basis_adjst=get_funcbasis_multivariate(x=test_points_adjst, L=L_adjst, type=method_args["basis_type"])
basis_temp=get_funcbasis_multivariate(x=test_points_adjst_temp, L=L_adjst, type=method_args["basis_type"])

# Fit DecoR without adjustement for the temperature
estimates_decor=get_results(x=x_norm, y=y, **method_args, basis=cosine_basis(n-1), L=L, K=K, lmbd=0.001)
y_est=basis @ estimates_decor["estimate"]
ci=get_conf(x=test_points, **estimates_decor, L=L, alpha=0.95, basis_type=method_args["basis_type"])

# Fit DecoR with adjustement for the temperature
estimates_decor_adjst=get_results(x=X, y=y, **method_args, basis=cosine_basis(n-1), L=L_adjst, K=K, lmbd=10**(-4))
y_adjst=basis_adjst @ estimates_decor_adjst["estimate"]
ci_adjst_help=conf_help(**estimates_decor_adjst, L=L_adjst, alpha=0.95)
H=basis_adjst[:,1:(L_adjst[0]+1)]@(ci_adjst_help['H'])[1:(L_adjst[0]+1), :]
sigma=ci_adjst_help['sigma']*np.sqrt(np.diag(H@H.T))
ci_adjst=np.stack((y_adjst-ci_adjst_help['qt']*sigma, y_adjst+ci_adjst_help['qt']*sigma)).T



# Regularized
Lmbd_min=10**(-8)       # smallest regularization parameter lambda to be considered
Lmbd_max=10**(1)        # largest regularization paramter lambda to be considered
n_lmbd=100              # number of lambda to test
L_cv=np.array([25, 25]) # number of coefficient for the reuglarized torrent
B=200                   # number of sample to draw for the bootstrap                                                                        
Lmbd=np.array([np.exp(i/n_lmbd*(np.log(Lmbd_max)-np.log(Lmbd_min))+np.log(Lmbd_min)) for i in range(0, n_lmbd)])      # grid of regularization paramters   

# Compute the basis and regularization matrix K for the smoothness penalty
basis_cv=get_funcbasis_multivariate(x=test_points_adjst, L=L_cv, type=method_args["basis_type"])
diag=np.concatenate((np.array([0]), np.array([i**4 for i in range(1,L_cv[0]+1)]),  np.array([i**4 for i in range(1, L_cv[1]+1)])))
K=np.diag(diag)

err_m= np.full([n_lmbd], float(0))
err_sd= np.full([n_lmbd], float(0))

# Basis expansion
basis= cosine_basis(n-1)
R=get_funcbasis_multivariate(x=X, L=L_cv)
tranformed={ 'xn': basis.T @ R/ n, 'yn' : basis.T @ y / n}
"""
# Compute the estimator of DecoR and the regulaized DecoR
for i in tqdm(range(n_lmbd)):
    err_b=err_boot(transformed=tranformed, a=0.9, lmbd=Lmbd[i], K=K, B=B)
    err_m[i]= sum(err_b['err_m'])/B
    err_sd[i]= np.linalg.norm(err_b['err_m']-err_m[i])/(B-1)

# Get lambda minimizing the estimated error
ind=np.argmin(err_m)
lmbd_min=Lmbd[ind]

# Compute the indices minimizing the estimated perdiction error
lmbd_1se=Lmbd[min(np.arange(ind,n_lmbd)[err_m[ind:n_lmbd]>err_m[ind]+err_sd[ind]])]
print(lmbd_1se)
"""

# Run Torrent and compute the error
estimates_reg = get_results(x=X, y=y, basis=basis, a=0.9, method="torrent_reg", basis_type=method_args["basis_type"], L=L_cv, K=K, lmbd=0.00001)
y_tor=basis_cv @ estimates_reg["estimate"]
y_adjst=np.ndarray((n_x, 1), buffer=y_tor)

ci_adjst_help=conf_help(**estimates_reg, L=L_cv, K=K, lmbd=0.00001, alpha=0.9)
H=basis_cv[:,1:(L_cv[0]+1)]@(ci_adjst_help['H'])[1:(L_cv[0]+1), :]
sigma=ci_adjst_help['sigma']*np.sqrt(np.diag(H@H.T))
sigma=np.ndarray((n_x, 1), buffer=sigma)
ci_adjst=np.stack((y_adjst-ci_adjst_help['qt']*sigma, y_adjst+ci_adjst_help['qt']*sigma)).T
ci_adjst=ci_adjst.reshape((n_x, 2))



# Fit benchmark for comparison and to make the confounding visable    
if bench_mark=="spline":
    gam = GAM(s(0)+intercept, lam=5).fit(np.reshape(X.T, (-1,2)), y)
    y_bench=gam.predict(test_points_adjst.T)
    ci_bench=gam.confidence_intervals(test_points_adjst.T, width=0.95)
else:
    estimates_fourier= get_results(x=x_norm, y=y, basis=cosine_basis(n-2), method="ols", L=L, basis_type=method_args["basis_type"], a=method_args["a"])
    y_bench=basis @ estimates_fourier["estimate"]
    ci_bench=get_conf(x=test_points, **estimates_fourier, alpha=0.95, basis_type=method_args["basis_type"])


# ----------------------------------
# plotting
# ----------------------------------

test_ozone=(test_points)*(x_max-x_min)+x_min

# Compute estimate from Bhaskaran et al. 2013
y_ref=np.exp(0.0007454149*test_ozone)*np.mean(y)
y_ref_l=np.exp(0.00042087681*test_ozone)*np.mean(y)
y_ref_u=np.exp(0.0010698931*test_ozone)*np.mean(y)

# Plot the difference estimations
plt.scatter(x=x, y=y, color='w', edgecolors="gray", s=4) 
plt.plot(test_ozone, y_bench, '-', color=ibm_cb[4], linewidth=1.5)
plt.plot(test_ozone, y_adjst, '-', color=ibm_cb[1], linewidth=1.5)
plt.plot(test_ozone, y_ref, '--', color=ibm_cb[2], linewidth=1)

# Plot confidence intervals
plt.fill_between(test_ozone, y1=ci_bench[:, 0], y2=ci_bench[:, 1], color=ibm_cb[4], alpha=0.4)
plt.fill_between(test_ozone, y1=ci_adjst[:, 0], y2=ci_adjst[:, 1], color=ibm_cb[1], alpha=0.4)
plt.fill_between(test_ozone, y1=y_ref_l, y2=y_ref_u, color=ibm_cb[2], alpha=0.4)

def get_handles():
    point_1 = Line2D([0], [0], label='Observations', marker='o', mec="gray", markersize=3, linestyle='')
    point_2= Line2D([0], [0], label='Bhaskaran et al.', color=ibm_cb[2], marker='', mec="black", markersize=3, linestyle='--')
    point_3 = Line2D([0], [0], label="DecoR" , color=ibm_cb[1], linestyle='-')
    point_4= Line2D([0], [0], label="GAM" , color=ibm_cb[4], linestyle='-')
    return [point_1,  point_2, point_3, point_4]

# Labeling
plt.xlabel("Ozone ($\mu g/m^3$)")
plt.ylabel("# Deaths")
plt.title("Influence of Ozone on Health")
plt.legend(handles=get_handles(), loc="upper left")
plt.grid(linestyle='dotted')
plt.tight_layout()
plt.show()


# ----------------------------------
# Plot the estimated outliers
# ----------------------------------

inl=estimates_decor_adjst["inliers"]
out=np.delete(np.arange(0,n), list(inl))
freq_rem=(out+0.5)/(2*n*24*3600)*10**6        #Conver to mikrohertz
plt.hist(freq_rem,  color=ibm_cb[0], edgecolor='k', alpha=0.6, bins=15)
plt.xlabel("Frequency ($\mu Hz$)")
plt.ylabel("Count")
plt.title("Histogramm of Excluded Frequencies")
plt.tight_layout()
plt.show()


# ----------------------------------
# Plot the influence of temperature on #death
# ----------------------------------

y_temp=basis_temp @ estimates_decor_adjst["estimate"]
test_temp=(test_points)*(t_max-t_min)+t_min

# compute the confidence interval
ci_adjst=np.stack((y_adjst-ci_adjst_help['qt']*sigma, y_adjst+ci_adjst_help['qt']*sigma)).T
H=basis_temp[:,(L_adjst[0]+1):(L_adjst[1]+L_adjst[0]+1)]@(ci_adjst_help['H'])[(L_adjst[0]+1):(L_adjst[0]+L_adjst[1]+1), :]
sigma=ci_adjst_help['sigma']*np.sqrt(np.diag(H@H.T))
ci_temp=np.stack((y_temp-ci_adjst_help['qt']*sigma, y_temp+ci_adjst_help['qt']*sigma)).T

plt.scatter(x=temp, y=y, marker='o', color='w', edgecolors="gray", s=5) 
plt.plot(test_temp, y_temp, '-', color=ibm_cb[2], linewidth=1.5)
plt.fill_between(test_temp, y1=ci_temp[:, 0], y2=ci_temp[:, 1], color=ibm_cb[2], alpha=0.3)

plt.grid(linestyle='dotted')
plt.ylabel("# Deaths")
plt.xlabel("temperature ($C^\circ$)")
plt.title("Influence of Temperature")
plt.tight_layout()
plt.show()


# ----------------------------------
# Comparing no-/adjustment to temperature
# ----------------------------------

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
