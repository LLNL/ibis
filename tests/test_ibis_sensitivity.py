from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import sys
import unittest
import warnings
import numpy as np
import scipy.stats as sts
from sklearn.gaussian_process import GaussianProcessRegressor
from ibis import sensitivity
from trata.sampler import OneAtATimeSampler, MorrisOneAtATimeSampler, SobolIndexSampler
import imageio
from skimage import color
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)


class TestUQMethods(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.baseline_dir = os.path.join(self.test_dir, 'baseline')
        self.X = np.array([[ 1.24137187,  8.21363941,  2.35916309, 12.98515141],
                           [ 3.97871385,  3.4634106 ,  1.81173418, 16.90803561],
                           [ 0.97135657,  8.55927091,  2.25817247, 14.90514011],
                           [ 4.70231176,  6.05691665,  2.97592548, 14.26088417],
                           [ 3.13157672,  2.56754092,  1.55154357, 11.08836465],
                           [ 2.85627701,  2.87869049,  2.94469817, 19.18005166],
                           [ 1.37481574,  6.34074534,  1.46366104,  9.98052500],
                           [ 4.86096237,  5.81760855,  2.36498487,  7.38065248],
                           [ 4.52033545,  4.11494366,  2.28123717, 19.63653232],
                           [ 0.66536309,  9.65189302,  2.84928394, 17.06356271],
                           [ 3.66626863,  5.9997253 ,  1.28051022, 13.39896002],
                           [ 2.3871072 ,  8.39860732,  1.37098550,  1.60682375],
                           [ 3.78112278,  3.30522716,  1.75767657,  1.37346581],
                           [ 1.83741247,  5.43530656,  1.05107072, 17.95442797],
                           [ 0.40752001,  6.91318904,  2.56303153,  9.19703154],
                           [ 2.57605785,  5.25440095,  2.41097409, 14.56978355],
                           [ 1.60637698,  7.39148000,  2.91944105,  6.82215659],
                           [ 1.58410412,  4.35974523,  1.93690498, 15.59792770],
                           [ 3.06926337,  2.13072566,  1.58920638,  8.81701752],
                           [ 1.14459004,  9.17563551,  1.22460758,  5.62802017],
                           [ 0.73623405,  9.85305137,  2.60722961,  3.16571594],
                           [ 4.05160946,  2.36922940,  2.46975318, 17.51093879],
                           [ 2.74573322,  7.04975780,  1.89461927,  5.33672380],
                           [ 3.48083622,  8.86717767,  1.27107048,  4.40735748],
                           [ 4.63162845,  7.13518101,  1.71173291,  9.38052265],
                           [ 0.58437871,  4.94494384,  1.42537214, 18.37452804],
                           [ 4.34892913,  4.00615419,  2.15042565, 16.05864968],
                           [ 0.15401196,  5.62393172,  2.18552212, 19.39526414],
                           [ 3.58261281,  4.44959386,  1.78620536, 18.54254783],
                           [ 0.25216043,  7.89461660,  2.64001915,  1.89825371],
                           [ 2.21425993,  6.71946639,  1.66125852, 12.49767619],
                           [ 3.33022636,  7.55259835,  1.96714155, 10.73079062],
                           [ 1.78070227,  8.66862129,  1.15076535, 13.90328880],
                           [ 2.63194950,  6.57974458,  2.80100101,  8.55532497],
                           [ 1.40212257,  9.03964998,  2.52263480,  6.17769319],
                           [ 2.94824458,  2.98554354,  1.85305056,  2.77477712],
                           [ 2.00260263,  3.89112673,  2.06335720, 11.60092112],
                           [ 3.25055741,  7.96839973,  1.16402311, 11.89907136],
                           [ 3.89180300,  2.65694748,  2.10007693,  7.64660330],
                           [ 4.42069661,  9.40580866,  1.63203985,  4.70493177],
                           [ 4.99257328,  3.71417673,  2.72281479,  7.93113303],
                           [ 4.12773230,  3.15814981,  1.03355614, 12.02374559],
                           [ 1.01760621,  9.31162651,  1.34761105, 10.15156742],
                           [ 1.99903012,  2.16408446,  2.68496630,  4.99197010],
                           [ 2.16926506,  4.70855981,  1.10134380,  3.84728350],
                           [ 0.07521400,  7.65809221,  2.51377788,  6.56267953],
                           [ 0.81162166,  4.84800917,  2.02469379, 16.41846506],
                           [ 0.30860007,  5.15512893,  1.51978080,  2.16839547],
                           [ 2.46656246,  6.25639849,  2.77336732, 15.21057963],
                           [ 4.23295714,  9.71703376,  2.23377103,  3.42678347]])
        def hill(x, a, b, c):
            return a * (x ** c) / (x ** c + b ** c)
        self.Y = hill(*self.X.T).reshape(-1, 1)
        self.input_names = ['x', 'a', 'b', 'c']
        self.ranges = np.array([[0, 5], [2, 10], [1, 3], [1, 20]])
        self.default = np.array([1, 5, 2, 5])
        self.output_names = ['y']
        self.seed = 3
        self.min_score = 0.9
        self.surrogate = GaussianProcessRegressor().fit(self.X, self.Y)

    def tearDown(self):
        if os.path.isfile('test_model'):
            os.remove('test_model')

    def test_oat_effects(self):
        OAT_samples = OneAtATimeSampler.sample_points(box=self.ranges,
                                                      default=self.default,
                                                      do_oat=True,
                                                      use_default=True)
        OAT_response = OAT_samples.sum(axis=1)**2
        effects = sensitivity.one_at_a_time_effects(OAT_samples, OAT_response)

        expected_effects = [[25., 23., 25., 22.], [30., 31., 27., 41.]]

        np.testing.assert_array_almost_equal(expected_effects, effects)

    def test_moat_effects(self):
        MOAT_samples = MorrisOneAtATimeSampler.sample_points(box=self.ranges,
                                                             num_paths=10,
                                                             seed=3)
        MOAT_response = MOAT_samples.sum(axis=1)**2
        m_effects = sensitivity.morris_effects(MOAT_samples, MOAT_response)

        expected_m_effects = [[40.55155969, 35.56105888, 33.46257093, 28.67382446],
                              [60.08594273, 34.63303031, 60.9013206,  46.74642484],
                              [39.22990622, 48.39725841, 40.08809413, 42.40868939],
                              [36.83668827, 35.56724055, 36.05495067, 46.46449389],
                              [45.57264425, 47.64336072, 63.77872573, 57.21152416],
                              [43.98668581, 44.43441978, 55.12472458, 50.43371629],
                              [63.6991378,  65.61514284, 65.62170232, 65.59540061],
                              [24.95859767, 35.96762307, 29.15773975, 38.54903518],
                              [59.04113187, 51.55765365, 51.10504083, 57.44843722],
                              [32.86249357, 49.73056039, 32.98620264, 42.01241613]]

        np.testing.assert_array_almost_equal(expected_m_effects, m_effects)

    def test_sobol_indices(self):
        sobol_samples = SobolIndexSampler.sample_points(num_points=3,
                                                        box=self.ranges,
                                                        seed=3)
        sobol_response = (sobol_samples.sum(axis=1)**2).reshape(-1, 1)

        first_order, total_order = sensitivity.sobol_indices(feature_data=sobol_samples,
                                                             response_data=sobol_response)
        first_expected = np.array([0.09444798, 0.07956545, 0.13519539, 0.11711406])
        total_expected = np.array([0.79952816, 0.76486211, 0.9250325,  0.84982509])

        np.testing.assert_array_almost_equal(first_expected, first_order)
        np.testing.assert_array_almost_equal(total_expected, total_order)

    def test_f_score(self):
        fig, ax = plt.subplots(1, 1)
        sensitivity.f_score_plot([ax], self.X, self.Y, ['x', 'a', 'b', 'c'], ['y'],
                                 degree=2, use_p_value=False)
        plt.savefig("out_fscore")
        plot_expected = "f_score_plot.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)

        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_fscore.png")[:, :, :3])

        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

    def test_f_p_score(self):
        fig, ax = plt.subplots(1, 1)
        sensitivity.f_score_plot([ax], self.X, self.Y, ['x', 'a', 'b', 'c'], ['y'],
                                 degree=2, use_p_value=True)
        plt.savefig("out_fp_score")
        plot_expected = "f_p_score_plot.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)

        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_fp_score.png")[:, :, :3])

        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

    def test_mutual_info_score(self):
        fig, ax = plt.subplots(1, 1)
        sensitivity.mutual_info_score_plot([ax], self.X, self.Y, self.input_names,
                                                self.output_names, n_neighbors=2)
        plt.savefig("out_mutual_info_score")
        plot_expected = "mutual_info_score_plot.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)

        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_mutual_info_score.png")[:, :, :3])

        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

    def test_pce_score(self):
        fig, ax = plt.subplots(1, 1)
        sensitivity.pce_score_plot([ax], self.X, self.Y, self.input_names, self.output_names,
                                   self.ranges, degree=2, model_degrees=2)
        plt.savefig("out_pce_score")
        plot_expected = "pce_score_plot.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)

        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_pce_score.png")[:, :, :3])

        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

    def test_lasso_path(self):
        fig, ax = plt.subplots(1, 1)
        sensitivity.lasso_path_plot([ax], self.X, self.Y, self.input_names, self.output_names, degree=1)
        plt.savefig("out_lasso_path")
        plot_expected = "lasso_path_plot.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)

        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_lasso_path.png")[:, :, :3])

        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

    def test_sensitivity_plot(self):
        fig, ax = plt.subplots(1, 4)
        sensitivity.sensitivity_plot(ax, self.surrogate, self.input_names, self.output_names,
                                     self.ranges, num_plot_points=10, num_seed_points=2,
                                     seed=3)
        plt.savefig("out_sensitivity")
        plot_expected = "sensitivity_plot.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)

        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_sensitivity.png")[:, :, :3])

        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

    def test_f_score_rank_plot(self):
        fig, ax = plt.subplots(1, 1)
        sensitivity.f_score_rank_plot(ax, self.X, self.Y, self.input_names, self.output_names,
                                      degree=2, interaction_only=True, use_p_value=False)
        plt.savefig("out_f_score_rank_io")
        plot_expected = "f_score_rank_plot_io.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)
        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_f_score_rank_io.png")[:, :, :3])
        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

        sensitivity.f_score_rank_plot(ax, self.X, self.Y, self.input_names, self.output_names,
                                      degree=2, interaction_only=True, use_p_value=True)
        plt.savefig("out_f_score_rank_io_pval")
        plot_expected = "f_score_rank_plot_io_pval.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)
        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_f_score_rank_io_pval.png")[:, :, :3])
        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

        sensitivity.f_score_rank_plot(ax, self.X, self.Y, self.input_names, self.output_names,
                                      degree=2, interaction_only=False, use_p_value=False)
        plt.savefig("out_f_score_rank")
        plot_expected = "f_score_rank_plot.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)
        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_f_score_rank.png")[:, :, :3])
        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

        sensitivity.f_score_rank_plot(ax, self.X, self.Y, self.input_names, self.output_names,
                                      degree=2, interaction_only=False, use_p_value=True)
        plt.savefig("out_f_score_rank_pval")
        plot_expected = "f_score_rank_plot_pval.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)
        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_f_score_rank_pval.png")[:, :, :3])
        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

    def test_mutual_info_rank_plot(self):
        fig, ax = plt.subplots(1, 1)
        sensitivity.mutual_info_rank_plot(ax, self.X, self.Y, self.input_names, self.output_names,
                                          n_neighbors=2)
        plt.savefig("out_mutual_info_rank")
        plot_expected = "mutual_info_rank_plot.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)

        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_mutual_info_rank.png")[:, :, :3])

        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

    def test_pce_rank_plot(self):
        fig, ax = plt.subplots(1, 1)
        sensitivity.pce_rank_plot(ax, self.X, self.Y, self.input_names, self.output_names,
                                  self.ranges, degree=2, model_degrees=2)
        plt.savefig("out_pce_rank")
        plot_expected = "pce_rank_plot.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)

        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_pce_rank.png")[:, :, :3])

        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

    def test_f_score_network_plot(self):
        fig, ax = plt.subplots(1, 2)
        sensitivity.f_score_network_plot(ax, self.X, self.Y, self.input_names, self.output_names,
                                         degree=3)
        plt.savefig("out_f_score_network")
        plot_expected = "f_score_network_plot.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)

        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_f_score_network.png")[:, :, :3])

        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > 0.8

    def test_pce_network_plot(self):
        fig, ax = plt.subplots(1, 2)
        sensitivity.pce_network_plot(ax, self.X, self.Y, self.input_names, self.output_names,
                                     self.ranges, degree=3, model_degrees=3)
        plt.savefig("out_pce_network")
        plot_expected = "pce_network_plot.png"
        base_dir = os.path.join(self.baseline_dir, plot_expected)

        base_img = color.rgb2gray(imageio.v2.imread(base_dir)[:, :, :3])
        out_img = color.rgb2gray(imageio.v2.imread("out_pce_network.png")[:, :, :3])

        score = structural_similarity(base_img, out_img, data_range=1.0)
        assert score > self.min_score

if __name__ == '__main__':
    unittest.main()