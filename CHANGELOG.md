# 1.1.0

# Release Notes

* [1.1.0](#110-release)


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