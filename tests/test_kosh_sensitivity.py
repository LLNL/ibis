from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import sys
import unittest
import pytest
import warnings
import numpy as np
import kosh
import h5py
import scipy.stats as sts
from sklearn.gaussian_process import GaussianProcessRegressor
from ibis import mcmc
from ibis import sensitivity
from ibis import kosh_operators
from trata import OneAtATimeSampler, MorrisOneAtATimeSampler


def test_kosh_oat_effects():

	# Hill function to get output data
	def hill(x, a, b, c):
		return a * (x ** c) / (x ** c + b ** c)

	ranges = np.array([[0, 5], [2, 10], [1, 3], [1, 20]])
	default = np.array([1, 5, 2, 5])

	OAT_samples = OneAtATimeSampler.sample_points(box=ranges, default=default, do_oat=True, use_default=True)
	OAT_response = hill(*OAT_samples.T)

	fileName = "mytestfile.hdf5"
	with h5py.File(fileName, "w") as f:
		f.create_dataset("inputs", data=OAT_samples)
		f.create_dataset("outputs", data=OAT_response)
		f.close()

	store = kosh.connect("temp_testing.sql")
	dataset = store.create("uq_data")
	dataset.associate([fileName], 'hdf5')

	oat_effects_actual = kosh_operators.KoshOneAtATimeEffects(dataset['inputs'],
								   		   	   				  input_names=['x','a','b','c'],
								   		   	   				  outputs=dataset['outputs'],
								   		   	   				  output_name=['response'],
								   		   	   				  method='OAT')[:]
	oat_effects_expected = np.array([[ 0.15151515,  0.03030303, -2.34848485, -0.37878788],
       							 	 [ 1.19945096,  0.03030303, -0.13102335, -0.01010069]])

    np.testing.assert_array_almost_equal(oat_effects_actual[name],
                                     	 oat_effects_expected[name])

    moat_effects_actual = kosh_operators.KoshOneAtATimeEffects(dataset['inputs'],
    														   input_names=['x','a','b','c'],
								   		   	   				   outputs=dataset['outputs'],
								   		   	   				   output_name=['response'],
								   		   	   				   method='MOAT')[:]
    moat_effects_expected = np.array([[ 2.92272041e+00,  7.36863527e-01, -7.36586718e-01, -1.01377959e-05],
       								  [ 2.30849124e+00,  7.20432683e-01, -6.02100069e-01, 6.79781008e-01],
       								  [ 4.40279245e-01,  1.22311121e-12, -1.66556852e+00, -1.19409343e-01],
       								  [ 2.62998595e+00,  1.17120779e-03, -5.93357821e-02, 8.84563122e-05],
							          [ 1.27781293e-02,  9.99979636e-01, -6.31003148e-02, 1.38378038e-03],
							          [ 1.37807640e+00,  3.24550037e-02, -5.01728364e-01, 3.76985410e-01],
							          [ 4.34892068e-02,  8.84980968e-01, -1.01245616e+00, 3.20322500e-02],
							          [ 2.06498306e+00,  4.37000912e-01, -2.13248203e+00, 7.72911783e-03],
							          [ 1.53018278e-01,  3.22836599e-02, -2.69962287e-01, -4.80432643e-12],
							          [ 1.44040399e+00,  9.96666152e-01, -1.35434979e-16, -7.72119504e-18]])
    np.testing.assert_array_almost_equal(moat_effects_actual[name],
                                     	 moat_effects_expected[name])

    def test_kosh_sensitivity_plots():

    	# Hill function to get output data
		def hill(x, a, b, c):
			return a * (x ** c) / (x ** c + b ** c)

		ranges = np.array([[0, 5], [2, 10], [1, 3], [1, 20]])
		default = np.array([1, 5, 2, 5])

		LHC_samples = LatinHyperCubeSampler.sample_points(box=ranges, num_points=50).astype('float')
		LHC_response = hill(*LHC_samples.T)

		surrogate_model = GaussianProcessRegressor().fit(LHC_samples, LHC_response)

		fileName = "mytestfile.hdf5"
		with h5py.File(fileName, "w") as f:
			f.create_dataset("inputs", data=LHC_samples)
			f.create_dataset("outputs", data=LHC_response)
			f.close()

		store = kosh.connect("temp_testing.sql")
		dataset = store.create("uq_data")
		dataset.associate([fileName], 'hdf5')


		result1 = kosh_operators.KoshSensitivityPlots(dataset['inputs'],
                                             method='lasso',
                                             input_names=['x','a','b','c'],
                                             outputs=dataset['outputs'],
                                             output_names=['response'],
                                             degree=1)[:]

		result2 = kosh_operators.KoshSensitivityPlots(dataset['inputs'],
		                                             method='sensitivity',
		                                             surrogate_model=surrogate_model,
		                                             input_names=['x','a','b','c'],
		                                             outputs=dataset['outputs'],
		                                             output_names=['response'],
		                                             input_ranges=ranges,
		                                             num_plot_points=10,
		                                             num_seed_points=2)[:]

		result3 = kosh_operators.KoshSensitivityPlots(dataset['inputs'],
		                                             method='f_score',
		                                             input_names=['x','a','b','c'],
		                                             outputs=dataset['outputs'],
		                                             output_names=['response'])[:]

		result4 = kosh_operators.KoshSensitivityPlots(dataset['inputs'],
		                                             method='mutual_info_score',
		                                             input_names=['x','a','b','c'],
		                                             outputs=dataset['outputs'],
		                                             output_names=['response'],
		                                             n_neighbors=3)[:]

		result5 = kosh_operators.KoshSensitivityPlots(dataset['inputs'],
		                                             method='pce_score',
		                                             input_names=['x','a','b','c'],
		                                             outputs=dataset['outputs'],
		                                             output_names=['response'],
		                                             input_ranges=ranges,
		                                             degree=1,
		                                             model_degrees=1)[:]

		result6 = kosh_operators.KoshSensitivityPlots(dataset['inputs'],
		                                             method='f_score_rank',
		                                             input_names=['x','a','b','c'],
		                                             outputs=dataset['outputs'],
		                                             output_names=['response'],
		                                             degree=1)[:]

		result7 = kosh_operators.KoshSensitivityPlots(dataset['inputs'],
		                                             method='mutual_info_rank',
		                                             input_names=['x','a','b','c'],
		                                             outputs=dataset['outputs'],
		                                             output_names=['response'],
		                                             n_neighbors=3)[:]

		result8 = kosh_operators.KoshSensitivityPlots(dataset['inputs'],
		                                             method='pce_rank',
		                                             input_names=['x','a','b','c'],
		                                             outputs=dataset['outputs'],
		                                             output_names=['response'],
		                                             degree=1,
		                                             model_degrees=1)[:]

		result9 = kosh_operators.KoshSensitivityPlots(dataset['inputs'],
		                                             method='f_score_network',
		                                             input_names=['x','a','b','c'],
		                                             outputs=dataset['outputs'],
		                                             output_names=['response'],
		                                             degree=1)[:]

		result10 = kosh_operators.KoshSensitivityPlots(dataset['inputs'],
		                                             method='pce_score_network',
		                                             input_names=['x','a','b','c'],
		                                             outputs=dataset['outputs'],
		                                             output_names=['response'],
		                                             degree=1)[:]




	