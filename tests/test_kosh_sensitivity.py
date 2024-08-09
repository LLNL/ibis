import os
import numpy as np
import kosh
import h5py
from ibis import kosh_operators
from trata.sampler import OneAtATimeSampler, MorrisOneAtATimeSampler


def test_kosh_oat_effects():

    # Hill function to get output data
    def hill(x, a, b, c):
        return a * (x ** c) / (x ** c + b ** c)

    ranges = np.array([[0, 5], [2, 10], [1, 3], [1, 20]])
    default = np.array([1, 5, 2, 5])

    OAT_samples = OneAtATimeSampler.sample_points(box=ranges, default=default, do_oat=True, use_default=True)
    OAT_response = hill(*OAT_samples.T)

    MOAT_samples = MorrisOneAtATimeSampler.sample_points(box=ranges, num_paths=10, seed=5)
    MOAT_response = hill(*MOAT_samples.T)

    fileName = "mytestfile2.hdf5"
    with h5py.File(fileName, "w") as f:
        f.create_dataset("inputs", data=OAT_samples)
        f.create_dataset("outputs", data=OAT_response)
        f.create_dataset("minputs", data=MOAT_samples)
        f.create_dataset("moutputs", data=MOAT_response)
        f.close()

    store = kosh.connect("sen_testing.sql")
    dataset = store.create("uq_data")
    dataset.associate([fileName], 'hdf5')

    oat_effects_actual = kosh_operators.KoshOneAtATimeEffects(dataset['inputs'],
                                                              input_names=['x', 'a', 'b', 'c'],
                                                              outputs=dataset['outputs'],
                                                              output_name=['response'],
                                                              method='OAT')[:]
    oat_effects_expected = np.array([[0.15151515,  0.03030303, -2.34848485, -0.37878788],
                                     [1.19945096,  0.03030303, -0.13102335, -0.01010069]])

    np.testing.assert_array_almost_equal(oat_effects_actual,
                                         oat_effects_expected)

    moat_effects_actual = kosh_operators.KoshOneAtATimeEffects(dataset['minputs'],
                                                               input_names=['x', 'a', 'b', 'c'],
                                                               outputs=dataset['moutputs'],
                                                               output_name=['response'],
                                                               method='MOAT')[:]
    moat_effects_expected = np.array([[4.28648011e-02,  9.94204532e-01, -1.92855818e-01, 1.16923509e-03],
                                      [1.68458287e+00,  7.37470159e-05, -4.19587808e-04, -7.34217555e-04],
                                      [2.15187225e-01,  9.97918098e-01, -1.13172239e-02, 1.74146824e-03],
                                      [1.91515777e+00,  7.44472168e-13, -4.50885545e+00, -2.26070938e-09],
                                      [2.95917398e-01,  9.10053999e-01, -2.06379426e-01, 2.20833245e-01],
                                      [5.22615836e+00,  6.85728269e-01, -6.63134512e+00, 1.61993499e-01],
                                      [2.81754178e-02,  9.98701060e-01, -1.20299115e-01, 4.81984047e-02],
                                      [2.61959302e+00,  3.30006477e-09, -6.95868204e-08, -1.33960173e-04],
                                      [1.18856883e+00,  1.90671956e-05, -2.48016858e-04, -2.63403096e-04],
                                      [8.82161209e-01,  5.11439321e-13, -2.36099761e-12, -1.93124765e-13]])

    np.testing.assert_array_almost_equal(moat_effects_actual,
                                         moat_effects_expected)

    os.remove(fileName)
    os.remove("sen_testing.sql")
