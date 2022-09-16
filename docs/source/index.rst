.. uq documentation master file, created by
   sphinx-quickstart on Thu Oct 11 15:43:19 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LLNL's UQ Methods Documentation!
==============================================

The LLNL Uncertainty Quantification (UQ) Pipeline, or UQP, is a Python-based scientific workflow system for running and analyzing concurrent UQ simulations on high-performance computers.  Using a simple, non-intrusive interface to simulation models, it provides the following capabilities:

* generating parameter studies
* generating one-at-a-time parameter variation studies
* sampling high dimensional uncertainty spaces
* generating ensemble of simulations leveraging LC's HPC resources
* analyzing ensemble of simulations output
* constructing surrogate models
* performing sensitivity studies
* performing statistical inferences
* estimating parameter values and probability distributions

The pipeline has been used for simulations in the domains of Inertial Confinement Fusion, National Ignition Facility experiments, climate, as well as other programs and projects.

The pipeline is composed of the following capabilities:

* sampling
* ensemble manager
* uncertainty quantification methods

We have made these capabilities available in individual software packages.

UQ Methods
===========

The `uq-methods` package is made up of 5 modules:
 * ``filter``
 * ``likelihoods``
 * ``mcmc``
 * ``mcmc_diagnostics``
 * ``plots``

The `uq-methods` package is designed to be used after a number of simulations have run to completion. 
This package is used to predict the results of future simulation runs.

The ``uq-methods`` package work with Python 2 and 3.
On LC RZ and CZ systems as an example, they are available at ``/collab/usr/gapps/uq/uq-methods``.
On LANL's Trinitite, they are available at ``/usr/projects/packages/uq-methods/``. Demo usage (on LC):

.. code:: python

        import sys
        sys.path.append("/collab/usr/gapps/uq/uq-methods")

        from uq_methods import mcmc

.. toctree::
   :maxdepth: 2
   :hidden:
   
   filters
   likelihoods
   mcmc
   plots

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`