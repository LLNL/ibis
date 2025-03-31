# Release Notes

* [1.1.1](#111-release)
* [1.1.0](#110-release)


## 1.1.1 Release

This release includes some bug fixes and an update to the basic descriptions.

### Improvements

* Updated descriptions for the main modules
* More explanation for descrepancy MCMC sampling

### Bug fixes

* KoshMCMC function only has experimental data coming from a Kosh store.
* Fixed loop to allow for any number of quantity of interest and experiments in the KoshMCMC function.
* Allow for the case of only one QOI in the sensitivity plot, variance_network_plot, and rank plot.

## 1.1.0 Release

This release introduces some new features

### New in this release

Added Kosh operators to Ibis to be able to use
IBIS UQ and sensitivity methods with Kosh datasets.

* KoshMCMC
* KoshOneAtATimeEffects
* KoshSensitivityPlots

A sobol_indices function has been added to the sensitivity module. It's meant
to be used with the SobolIndexSampler in the Trata sampler module or similar.