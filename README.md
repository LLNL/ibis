# UQ Methods

LLNL's UQ Methods is designed to be used after a number of simulations have run to completion and is used to predict the results of future simulation runs.


The **uq-methods** package contains 5 modules:
   - filter
   - likelihoods
   - mcmc
   - mcmc_diagnostics
   - plots


## Basic Installation

### via pip:

```bash
export UQ_METHODS_PATH = uq-methods                                  # `uq-methods` can be any name/directory you want
pip install virtualenv                                               # just in case
python3 -m virtualenv $UQ_METHODS_PATH   
source ${UQ_METHODS_PATH}/bin/activate
pip install "numpy>=1.15,<1.19" scikit-learn scipy matplotlib networkx
git clone https://github.com/LLNL/uq-methods
cd uq-methods
pip install .
```

### via conda:

```bash
conda create -n uq-methods -c conda-forge "python>=3.6" "numpy>=1.15,<1.19" scikit-learn scipy matplotlib networkx
conda activate uq-methods
git clone https://github.com/LLNL/uq-methods
cd uq-methods
pip install .
```
## Build Docs

### via pip:

```bash
pip install sphinx sphinx_rtd_theme
```
### via conda:

```bash
conda install -n uq-methods -c conda-forge sphinx sphinx_rtd_theme sphinx-autoapi nbsphinx
```

## Beefy Installation

### via pip:

```bash
export UQ_METHODS_PATH = uq-methods                               # `uq-methods` can be any name/directory you want
pip install virtualenv                                            # just in case
python3 -m virtualenv $UQ_METHODS_PATH   
source ${UQ_METHODS_PATH}/bin/activate
pip install "numpy>=1.15,<1.19" scikit-learn scipy matplotlib networkx six pip sphinx sphinx_rtd_theme ipython jupyterlab
git clone https://github.com/LLNL/uq-methods
cd uq-methods
pip install .
```
### via conda:

```bash
conda create -n uq-methods -c conda-forge "python>=3.6" "numpy>=1.15,<1.19" scikit-learn scipy matplotlib six pip networkx sphinx sphinx_rtd_theme sphinx-autoapi nbsphinx jupyterlab ipython ipywidgets nb_conda nb_conda_kernels 
conda activate uq-methods
git clone https://github.com/LLNL/uq-methods
cd uq-methods
pip install .
```

### Register your Python env via Jupyter:

```bash
python -m ipykernel install --user --name uq-methods --display-name "UQ Methods Environment"
```