# Universal Loss Reweighting to Balance Lesion Size Inequality in 3D Medical Image Segmentation
Code release

## Table of Contents
* [*iw* in a few lines of code](#*iw*-in-a-few-lines-of-code)
* [Requirements](#requirements)
* [Repository Structure](#repository-structure)
* [Experiment Reproduction](#experiment-reproduction)

## *iw* in a few lines of code
(TODO)

## Requirements
- [Python](https://www.python.org) (v3.6 or later)
- [Deep pipe](https://github.com/neuro-ml/deep_pipe) (commit: [4383211ea312c098d710fbeacc05151e10a27e80](https://github.com/neuro-ml/deep_pipe/tree/4383211ea312c098d710fbeacc05151e10a27e80))
- [imageio](https://pypi.org/project/imageio/) (v 2.8.0)
- [NiBabel](https://pypi.org/project/nibabel/) (v3.0.2)
- [NumPy](http://numpy.org/) (v1.17.0 or later)
- [OpenCV python](https://pypi.org/project/opencv-python/) (v4.2.0.32)
- [Pandas](https://pandas.pydata.org/) (v1.0.1 or later)
- [pdp](https://pypi.org/project/pdp/) (v 0.3.0)
- [pydicom](https://pypi.org/project/pydicom/) (v 1.4.2)
- [resource-manager](https://pypi.org/project/resource-manager/) (v 0.11.1)
- [SciPy library](https://www.scipy.org/scipylib/index.html) (v0.19.0 or later)
- [scikit-image](https://scikit-image.org) (v0.15.0 or later)
- [Simple ITK](http://www.simpleitk.org/) (v1.2.4)
- [torch](https://pypi.org/project/torch/) (v1.1.0 or later)
- [tqdm](https://tqdm.github.io) (v4.32.0 or later)

## Repository Structure
```
├── config
│   ├── assets
│   ├── exp_holdout
│   │   ├── lits
│   │   │   └── *.config
│   │   └── luna
│   │       └── *.config
│   └── exp_val
│       └── *.config
├── iw
│   ├── dataset
│   │   ├── lits.py
│   │   └── luna.py
│   ├── model
│   │   └── unet.py
│   ├── torch.py
│   ├── path.py (path_local.py)
│   └── ...
├── notebook
│   ├── data_preprocessing
│   │   ├── LITS_preprocessing.ipynb
│   │   ├── LUNA16_download.ipynb
│   │   └── LUNA16_preprocessing.ipynb
│   └── results
│       └── build_froc.ipynb
├── plots
│   └── froc.pdf
├── froc_data
│   └── ... (experiments structure)
├── model
│   └── [TODO]
└── README.md
```

Publicly available datasets could be downloaded and preprocessed
using notebooks in `notebook/data_preprocessing`. Also, resulting
plots could be build in `notebook/results`.

Inverse weighting is made via two parts in our lib:
- generating connected components for every ground truth. Could be found as
 function `load_cc` in `iw/dataset/lits.py` or `iw/dataset/luna.py`.
- creating weights for every incoming patch into loss function. Could be
found as function `cc2weights` in `iw/torch.py` .
These weights are used in `iwbce`, `iwdl`, `iwasl` and `iwfl` loss functions
in `iw/torch.py`.

All final experiments (for publicly available data) could be built via
configs in `config/exp_holdout`. To successfully process data and build-run
experiments one need to change core paths `iw/path.py`.

Validation experiments could be built via configs in `config/exp_val`. It
includes hyperparameters search for the Focal Loss. Results are presented in
`notebook/results/val_exps_visualization.ipynb`.

## Experiment Reproduction
To run a single experiment please follow the steps below:

First, the experiment structure must be created:
```
python -m dpipe build_experiment --config_path "$1" --experiment_path "$2"
```

where the first argument is a path to the `.config` file e.g., 
`"~/miccai2020paper1600/config/exp_holdout/luna/iwdl.config"`
and the second argument is a path to the folder where the experiment
structure will be organized, e.g.
`"~/miccai2020paper1600/froc_data/luna/iwdl"`.

Then, to run an experiment please go to the experiment folder inside the created structure:
```
cd ~/miccai2020paper1600/froc_data/luna/iwdl
```
and call the following command to start the experiment:
```
python -m dpipe run_experiment --config_path "../resources.config"
```
where `resources.config` is the general `.config` file of the experiment.
