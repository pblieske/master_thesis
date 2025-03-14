[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/pblieske/master_thesis/actions/workflows/python-app.yml/badge.svg)](https://github.com/pblieske/master_thesis/actions/workflows/python-app.yml)

# Inference of Nonlinear Causal Effects in Time Series in the Presence of Confounding


Master's Thesis write by Pio Blieske at the Seminar of Statistics at ETH Zürich under the supervision of Jonas Peters and Felix Schur. The repository was originally forked from https://github.com/fschur/robust_deconfounding.


## Abstract

Estimating causal effects between time series is of interest in many scientific fields such
as climate science, epidemiology, and economics, but remains challenging due to possible
confounding. This thesis focuses on the inference of nonlinear causal effects between two
time series in the presence of a third unobserved, confounding time series. We assume
sparsity of the confounder in the frequency domain, corresponding in applications, for
example, to a seasonal periodicity. By developing a new transformation for the data,
we leverage the sparsity assumption to reduce the confounding problem to an adversarial
outlier problem, a technique known as deconfounding by robust regression (DecoR). We
then use the robust regression algorithm Torrent to solve the adversarial outlier problem.
To improve the estimation accuracy, we extend Torrent to a regularized version, which
allows the incorporation of a smoothness penalty in DecoR, and provide upper bounds for
the estimation error. For two different asymptotic settings, we prove the consistency of
the nonlinear extensions of DecoR under suitable assumptions. We validate the nonlinear
extensions of DecoR by a simulation study on synthetic data. In addition, we demonstrate
its effectiveness with an application to a real-world example of environmental epidemiology.


## Reproducing the Results

For full support of all scripts in the repository, for instance to reproduce the experiments, further dependencies need
to be installed. 
To do so, please run in the main directory of this repository 
```bash
pip install -r requirements.txt
``` 
The structure of the repository consists mainly of two folders. DecoR and the robust regression algorithms can be found in the robust_deconfounding folder and can be installed as package as follows:
```bash
python setupy.py install
``` 
In the other folder, experiments, all the scripts for the simulations and the real-world data example can be found. For the consistency experiment, the plot is produced using a sperate script plot_consistency.py, for all the other experiment there is only one file for the simulation and plotting. The simulations are only rerun if the variable "run_exp" at the start of the file is set to "True", else the saved results are plotted. To run the a script e.g. use:
```bash
python experiments/ozone.py
``` 
All scripts contain a description of what they do at the beginning. The remaining folders are used for testing.