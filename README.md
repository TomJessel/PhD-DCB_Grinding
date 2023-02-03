# Acoustic-Emission
Acoustic-Emission is a repo for my PhD work.\
It includes code for processing and displaying AE and NC4 data. As well as 
using ML for tool condition monitoring.

## Repo Layout
1) [resources](resources)
   - [`experiment.py`](resources/experiment.py)
   - [`ae.py`](resources/ae.py)
   - [`nc4.py`](resources/nc4.py)
   - [`ml_mlp.py`](resources/ml_mlp.py)
   - [`surf_meas.py`](resources/surf_meas.py)
2) [ml](ml) 
3) [`testing_main.py`](testing_main.py)
4) [reference](reference)

## Usage
### [resources](resources)
Resources contains all the main files for processing experiment AE and NC4 
data. As well as classes for ML and Surface measurements.

### [ml](ml)
ML has .py and .ipynb files to use and display ML classes for the AE and 
NC4 data.\
[`ml_testing.ipynb`](ml/ml_testing.ipynb) creates and scores ML models via 
CV and validation sets.\
[`hparam_opt.py`](ml/hparam_opt.py) optimises a single architecture with 
gridsearch with given hparams.\
[`hparam_results.ipynb`](ml/hparam_results.ipynb) visualises the 
`hparam_opt.py` results from the tensorboard log files.\

### [testing_main](testing_main.py)
[`testing_main.py`](testing_main.py) has simple functions for checking 
acquired data during 
tool life tests. 

### [reference](reference)
Reference contains constant files for operation of other scripts. Including 
a .txt file for locating the experiment obj save locations.