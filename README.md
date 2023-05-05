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

## Getting Started

To get the latest public version:

```bash
pip install ibis
```

To get the latest stable from a cloned repo, simply run:

```bash
pip install .
```

Alternatively, add the path to this repo to your PYTHONPATH environment variable or in your code with:

```bash
import sys
sys.path.append(path_to_ibis_repo)
```
## Contact Info

IBIS maintainer can be reached at: eljurf1@llnl.gov

## Contributing

Contributions should be submitted as a pull request pointing to the develop branch, and must pass IBIS's CI process; to run the same checks locally, use:

```bash
pytest tests/test_*.py
```