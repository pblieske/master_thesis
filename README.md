[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Inference of Nonlinear Causal Effects in Time Series in the Presence of Confounding


> [!CAUTION]
> Under construction!

Master's Thesis write by Pio Blieske at the Seminar of Statistics at ETH Zürich under the supervision of Jonas Peters and Felix Schur. The repository was originally forked from https://github.com/fschur/robust_deconfounding.

## Abstract

Put abstract here.


## Reproducing the results
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
In the other folder, experiments, all the scripts for the simulations and the real-world data example can be found. For the consistency experiment, the plot is produced using a sperate script plot_consistency.py, for all the other experiment there is only one file for the simulation and plotting. The simulations are only rerun if the variable "run_exp" at the start of the file is set to True, else the saved results are plotted. To run the a script e.g. use:
```bash
python experiments/ozone.py
``` 
