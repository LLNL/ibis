<img align="left" width="75" height="75" src="./logo.png"> <br> 
# IBIS

LLNL's Interactive Bayesian Inference and Sensitivity, or IBIS, is designed to be used after a number of simulations have run to completion, to predict the results of future simulation runs.

Assessment of system performance variation induced by uncertain parameter values is referred to as uncertainty quantification (UQ). Typically, the Monte Carlo method is used to perform UQ by assigning probability distributions to uncertain input variables from which to draw samples in order to calculate corresponding output values using surrogate models. Based on the ensemble of output results, the output distribution should statistically describe the output's uncertainty.

Sensitivity analysis refers to the study of how uncertainty in the output of a mathematical model or system can be attributed to different sources of uncertainty in the inputs. In the data science space, sensitivity analysis is often called feature selection. 

In general, we have some function $`f`$ that we want to model. This is usually some sort of computer simulation where we vary a set of parameters $`X`$ to produce a set of outputs $`Y=f(X)`$.
We then ask the questions, "How does $`Y`$ change as $`X`$ changes?" and "Which parts of $`X`$ is $`Y`$ sensitive to?", this is often done so that we can choose to ignore the parameters of $`X`$ which don't affect $`Y`$ in subsequent analyses.

The IBIS package contains 7 modules:
   - filter
   - likelihoods
   - mcmc
   - mcmc_diagnostics
   - sensitivity
   - pce_model
   - plots

## Basic Installation

### via pip:

```bash
export IBIS_PATH = ibis                                  # `ibis` can be any name/directory you want
pip install virtualenv                                   # just in case
python3 -m virtualenv $IBIS_PATH   
source ${IBIS_PATH}/bin/activate
pip install numpy scikit-learn scipy matplotlib networkx
git clone https://github.com/LLNL/IBIS
cd ibis
pip install .
```

### via conda:

```bash
conda create -n ibis -c conda-forge "python>=3.6" numpy scikit-learn scipy matplotlib networkx
conda activate ibis
git clone https://github.com/LLNL/IBIS
cd ibis
pip install .
```
## Build Docs

### via pip:

```bash
pip install sphinx sphinx_rtd_theme
```
### via conda:

```bash
conda install -n ibis -c conda-forge sphinx sphinx_rtd_theme sphinx-autoapi nbsphinx
```

## Beefy Installation

### via pip:

```bash
export IBIS_PATH = ibis                               # `ibis` can be any name/directory you want
pip install virtualenv                                # just in case
python3 -m virtualenv $IBIS_PATH   
source ${IBIS_PATH}/bin/activate
pip install numpy scikit-learn scipy matplotlib networkx six pip sphinx sphinx_rtd_theme ipython jupyterlab pytest
git clone https://github.com/LLNL/IBIS
cd ibis
pip install .
```
### via conda:

```bash
conda create -n ibis -c conda-forge "python>=3.6" numpy scikit-learn scipy matplotlib six pip networkx sphinx sphinx_rtd_theme sphinx-autoapi nbsphinx jupyterlab ipython ipywidgets nb_conda nb_conda_kernels pytest
conda activate ibis
git clone https://github.com/LLNL/IBIS
cd ibis
pip install .
```

### Register your Python env via Jupyter:

```bash
python -m ipykernel install --user --name ibis --display-name "IBIS Environment"
```
