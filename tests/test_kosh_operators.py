from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import sys
import unittest
import warnings
import numpy as np
import kosh
import h5py
import scipy.stats as sts
from sklearn.gaussian_process import GaussianProcessRegressor
from ibis import kosh_operators

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TestUQMethods(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.46123442, 0.91202244, 0.92876321], [0.06385122, 0.08942266, 0.44186744],
                           [0.68961511, 0.52403073, 0.56833564], [0.85123686, 0.77737199, 0.52990221],
                           [0.28834075, 0.01743994, 0.36242905], [0.54587868, 0.41817851, 0.72421346],
                           [0.25357849, 0.22729639, 0.5369119], [0.15648554, 0.3817992, 0.08481071],
                           [0.06034016, 0.89202591, 0.68325111], [0.47304368, 0.85874444, 0.28670653],
                           [0.13013734, 0.68430436, 0.38675013], [0.42515756, 0.90183769, 0.834895],
                           [0.43438549, 0.56288698, 0.65607153], [0.39890529, 0.66302321, 0.72165207],
                           [0.72526413, 0.30656396, 0.98994899], [0.50290481, 0.02484515, 0.05371448],
                           [0.24154745, 0.19699787, 0.08694096], [0.86792345, 0.73439781, 0.94549644],
                           [0.66545947, 0.05688571, 0.34595029], [0.95990758, 0.05884415, 0.46791467],
                           [0.86474237, 0.79569641, 0.85598485], [0.63037856, 0.10142019, 0.61359389],
                           [0.1390893, 0.1522652, 0.62634887], [0.33668492, 0.36119699, 0.58089201],
                           [0.29715877, 0.73680818, 0.20449131], [0.55683808, 0.49606268, 0.30689581],
                           [0.61112437, 0.45046932, 0.72687226], [0.82401322, 0.0999469, 0.09535599],
                           [0.76943412, 0.13181057, 0.81715459], [0.6067913, 0.52855681, 0.73084772],
                           [0.77817408, 0.23410479, 0.77393847], [0.31784351, 0.55372617, 0.71227582],
                           [0.46758069, 0.35570418, 0.31543622], [0.06989688, 0.82159477, 0.3253991],
                           [0.30579717, 0.56461504, 0.07109011], [0.50532271, 0.57142318, 0.59031356],
                           [0.11022146, 0.1806901, 0.20185294], [0.37793607, 0.45846717, 0.40057469],
                           [0.891715, 0.51578042, 0.05885039], [0.5729882, 0.29217043, 0.12501581],
                           [0.90126537, 0.76969839, 0.52675768], [0.00216229, 0.14707118, 0.9368788],
                           [0.16484123, 0.55441898, 0.83753256], [0.56469525, 0.38370204, 0.65722823],
                           [0.8687188, 0.66432443, 0.67008946], [0.48001072, 0.50522088, 0.13284311],
                           [0.91704432, 0.99687862, 0.65933211], [0.75265367, 0.11150535, 0.9612883],
                           [0.39109998, 0.23905451, 0.6992524], [0.00245559, 0.07515066, 0.7427796],
                           [0.43617553, 0.81086171, 0.76694599], [0.05498618, 0.68288433, 0.05098977],
                           [0.92852732, 0.92922038, 0.07499123], [0.3563414, 0.0369639, 0.80971561],
                           [0.31104242, 0.26773013, 0.24337643], [0.40656828, 0.84523629, 0.92413601],
                           [0.2621117, 0.13541767, 0.13898699], [0.78952943, 0.27979129, 0.36594954],
                           [0.96398771, 0.39427822, 0.42041622], [0.06170219, 0.39562485, 0.0390669],
                           [0.07484891, 0.44352503, 0.86574964], [0.02119805, 0.08114133, 0.66240878],
                           [0.62832535, 0.74553018, 0.33435648], [0.27253955, 0.05851183, 0.9477553],
                           [0.17621574, 0.48891392, 0.08004835], [0.05899438, 0.49678554, 0.39423793],
                           [0.13625638, 0.90123555, 0.99555211], [0.82430987, 0.06799042, 0.98713305],
                           [0.23391724, 0.32835972, 0.11899672], [0.26675385, 0.49923745, 0.5294856],
                           [0.50285101, 0.75814327, 0.30801608], [0.97083411, 0.25323657, 0.91860817],
                           [0.50567205, 0.07012236, 0.12421462], [0.58163984, 0.34303427, 0.36467924],
                           [0.62053834, 0.542813, 0.77542096], [0.04564984, 0.57157144, 0.2524628],
                           [0.98461689, 0.06922946, 0.61206824], [0.18133769, 0.85259098, 0.2208197],
                           [0.02360491, 0.58486804, 0.88898217], [0.24356099, 0.78698977, 0.19156109],
                           [0.3374873, 0.3931525, 0.34168161], [0.04891735, 0.06757889, 0.76139633],
                           [0.19553807, 0.02900628, 0.58441379], [0.08725175, 0.47520548, 0.3877658],
                           [0.72472161, 0.462946, 0.39650086], [0.2661204, 0.82420122, 0.6588341],
                           [0.69032023, 0.53098725, 0.39433453], [0.11943751, 0.74554536, 0.87115827],
                           [0.18756975, 0.83759763, 0.44177224], [0.21552329, 0.82555553, 0.85337084],
                           [0.59029845, 0.40985213, 0.86169482], [0.22949626, 0.30654941, 0.28231961],
                           [0.17845353, 0.94908186, 0.56501311], [0.91970551, 0.04106241, 0.11949207],
                           [0.7979433, 0.50880488, 0.81055288], [0.81982103, 0.36466048, 0.13310552],
                           [0.37220176, 0.99673639, 0.39217999], [0.71401306, 0.82261441, 0.79515913],
                           [0.11756912, 0.45101294, 0.76186856], [0.93985828, 0.92252428, 0.12734155]])
        self.Y = np.stack([self.X.sum(axis=1) ** 2, np.sin(self.X.sum(axis=1))]).T
        self.input_names = ['input_A', 'input_B', 'input_C']
        self.output_names = ['output_A', 'output_B']
        self.seed = 15
        self.surrogate_model = {
            name: GaussianProcessRegressor().fit(self.X, self.Y[:, i])
            for i, name in enumerate(self.output_names)
        }
        self.s_model = GaussianProcessRegressor().fit(self.X, self.Y)
        self.ranges = [[0, 1], [0, 1], [0, 1]]
        self.flattened = False
        self.scaled = True
        fileName = "mytestfile.hdf5"
        with h5py.File(fileName, "w") as f:
            f.create_dataset("inputs", data=self.X)
            f.create_dataset("outputs", data=self.Y)
            f.close()

        self.store = kosh.connect("temp_testing.sql")
        self.dataset = self.store.create("uq_data")
        self.dataset.associate([fileName], 'hdf5')


    def tearDown(self):
        if os.path.isfile('test_model'):
            os.remove('test_model')

    @unittest.skipIf(sys.version_info[0] < 3, reason="Not supported for Python 2")
    def test_kosh_default_mcmc(self):
        result = kosh_operators.KoshMCMC(self.dataset["inputs"],
                                         method="default_mcmc",
                                         input_names=self.input_names,
                                         inputs_low=[0.0]*3,
                                         inputs_high=[1.0]*3,
                                         proposal_sigmas=[0.2]*3,
                                         prior=[sts.beta(2, 2).pdf]*3,
                                         outputs=self.dataset["outputs"],
                                         output_names=self.output_names,
                                         quantity='x',
                                         surrogate_model=self.surrogate_model,
                                         observed_values=[.5]*2,
                                         observed_std=[.1]*2,
                                         inputs=self.input_names,
                                         total_samples=10,
                                         burn=20,
                                         every=2,
                                         start={name: .5 for name in self.input_names},
                                         prior_only=True,
                                         seed=self.seed,
                                         flattened=False)[:]
        chains = result.get_chains(flattened=self.flattened, scaled=self.scaled)

        prior_expected = {'input_A': np.array([[0.32151482, 0.44159015, 0.28441874, 0.42174969, 0.3671779 ,
                                                0.33179506, 0.48374379, 0.511352  , 0.60738642, 0.60738642]]),
                          'input_B': np.array([[0.78009572, 0.8826035 , 0.8289768 , 0.73403566, 0.7457847 ,
                                                0.48535026, 0.30002295, 0.48387395, 0.39900237, 0.39900237]]),
                          'input_C': np.array([[0.45828809, 0.45416425, 0.41885536, 0.36999938, 0.33504304,
                                                0.3592561 , 0.26492496, 0.48845248, 0.49120407, 0.49120407]])}

        result2 = kosh_operators.KoshMCMC(self.dataset["inputs"],
                                          method="default_mcmc",
                                          input_names=self.input_names,
                                          inputs_low=[0.0]*3,
                                          inputs_high=[1.0]*3,
                                          proposal_sigmas=[0.2]*3,
                                          prior=[sts.beta(2, 2).pdf]*3,
                                          outputs=self.dataset["outputs"],
                                          output_names=self.output_names,
                                          quantity='x',
                                          surrogate_model=self.surrogate_model,
                                          observed_values=[.5]*2,
                                          observed_std=[.1]*2,
                                          inputs=self.input_names,
                                          total_samples=10,
                                          burn=20,
                                          every=2,
                                          start={name: .5 for name in self.input_names},
                                          prior_only=False,
                                          seed=self.seed,
                                          flattened=True)[:]
        chains2 = result2.get_chains(flattened=self.flattened, scaled=self.scaled)

        post_expected = {'input_A': np.array([[0.12877608, 0.24885141, 0.09168   , 0.09168   , 0.03786498,
                                               0.03786498, 0.13003829, 0.13003829, 0.22607271, 0.2609016 ]]),
                         'input_B': np.array([[0.14939956, 0.25190734, 0.19828064, 0.19828064, 0.41193175,
                                               0.41193175, 0.34268075, 0.34268075, 0.25780918, 0.10275209]]),
                         'input_C': np.array([[0.20644045, 0.20231661, 0.16700772, 0.16700772, 0.08575977,
                                               0.08575977, 0.08532267, 0.08532267, 0.08807426, 0.2397848 ]])}

        for name in self.input_names:
            np.testing.assert_array_almost_equal(chains[name],
                                                 prior_expected[name])
            np.testing.assert_array_almost_equal(chains2[name],
                                                 post_expected[name])


    def test_kosh_discrepancy_mcmc(self):
        start = {name: .5 for name in self.input_names}
        start['tau_x'] = .5
        result = kosh_operators.KoshMCMC(self.dataset["inputs"],
                                         method="discrepancy_mcmc",
                                         input_names=self.input_names,
                                         inputs_low=[0.0]*3,
                                         inputs_high=[1.0]*3,
                                         proposal_sigmas=[0.2]*3,
                                         prior=[sts.beta(2, 2).pdf]*3,
                                         outputs=self.dataset["outputs"],
                                         output_names=self.output_names,
                                         quantity='x',
                                         surrogate_model=self.surrogate_model,
                                         observed_values=[.5]*2,
                                         observed_std=[.1]*2,
                                         inputs=self.input_names,
                                         total_samples=10,
                                         burn=20,
                                         every=2,
                                         start=start,
                                         prior_only=True,
                                         seed=self.seed,
                                         flattened=False)[:]
        chains = result.get_chains(flattened=self.flattened, scaled=self.scaled)
        prior_expected = {'input_A': np.array([[0.46345716, 0.50974877, 0.63393836, 0.63393836, 0.71894928,
                                                0.66062354, 0.66062354, 0.73850457, 0.73850457, 0.49556849]]),
                          'input_B': np.array([[0.4838711 , 0.07577377, 0.06375146, 0.06375146, 0.17948177,
                                                0.30874394, 0.30874394, 0.53893049, 0.53893049, 0.49280753]]),
                          'input_C': np.array([[0.7840451 , 0.60787943, 0.43002658, 0.43002658, 0.27496949,
                                                0.57189731, 0.57189731, 0.61241226, 0.61241226, 0.48048973]]),
                          'tau_x': np.array([[0.13174745, 0.06946011, 0.09324664, 0.09324664, 0.13117427,
                                              0.21830206, 0.21830206, 0.19166014, 0.19166014, 0.15114434]])}

        result2 = kosh_operators.KoshMCMC(self.dataset["inputs"],
                                          method="discrepancy_mcmc",
                                          input_names=self.input_names,
                                          inputs_low=[0.0]*3,
                                          inputs_high=[1.0]*3,
                                          proposal_sigmas=[0.2]*3,
                                          prior=[sts.beta(2, 2).pdf]*3,
                                          outputs=self.dataset["outputs"],
                                          output_names=self.output_names,
                                          quantity='x',
                                          surrogate_model=self.surrogate_model,
                                          observed_values=[.5]*2,
                                          observed_std=[.1]*2,
                                          inputs=self.input_names,
                                          total_samples=10,
                                          burn=20,
                                          every=2,
                                          start=start,
                                          prior_only=False,
                                          seed=self.seed,
                                          flattened=True)[:]
        chains2 = result2.get_chains(flattened=self.flattened, scaled=self.scaled)
        post_expected = {'input_A': np.array([[0.69888725, 0.69888725, 0.82307684, 0.92350497, 0.92350497,
                                               0.65506841, 0.6660201 , 0.74390113, 0.74390113, 0.62464854]]),
                         'input_B': np.array([[0.35711477, 0.35711477, 0.34509246, 0.87210644, 0.87210644,
                                               0.78541696, 0.55307605, 0.78326259, 0.78326259, 0.59625489]]),
                         'input_C': np.array([[0.87951977, 0.87951977, 0.70166692, 0.25238694, 0.25238694,
                                               0.27001599, 0.50338092, 0.54389587, 0.54389587, 0.5978228 ]]),
                         'tau_x': np.array([[0.15326688, 0.15326688, 0.17705341, 0.21268524, 0.21268524,
                                             0.23775688, 0.29051215, 0.26387024, 0.26387024, 0.27889584]])}

        for name in start.keys():
            np.testing.assert_array_almost_equal(chains[name],
                                                 prior_expected[name])
            np.testing.assert_array_almost_equal(chains2[name],
                                                 post_expected[name])

    def test_kosh_sensitivity_plots(self):

        import matplotlib
        matplotlib.use("agg", force=True)
        import matplotlib.pyplot as plt

        methods = ["lasso", "sensitivity", "f_score", "mutual_info_score", "pce_score",
                   "f_score_rank", "mutual_info_rank", "pce_rank", "f_score_network",
                   "pce_network"]

        for m in methods:
            plot_name = f"output_A_output_B_{m}_plot.png"
            try:
                os.remove(plot_name)
            except BaseException:
                pass

        result1 = kosh_operators.KoshSensitivityPlots(self.dataset['inputs'],
                                                      method='lasso',
                                                      input_names=self.input_names,
                                                      outputs=self.dataset['outputs'],
                                                      output_names=self.output_names,
                                                      degree=1,
                                                      save_plot=True)[:]
        assert type(result1) == type(plt.figure())
        plt.close()

        result2 = kosh_operators.KoshSensitivityPlots(self.dataset['inputs'],
                                                      method='sensitivity',
                                                      surrogate_model=self.s_model,
                                                      input_names=self.input_names,
                                                      outputs=self.dataset['outputs'],
                                                      output_names=self.output_names,
                                                      input_ranges=self.ranges,
                                                      num_plot_points=10,
                                                      num_seed_points=2,
                                                      save_plot=True)[:]
        assert type(result2) == type(plt.figure())
        plt.close()

        result3 = kosh_operators.KoshSensitivityPlots(self.dataset['inputs'],
                                                      method='f_score',
                                                      input_names=self.input_names,
                                                      outputs=self.dataset['outputs'],
                                                      output_names=self.output_names,
                                                      save_plot=True)[:]
        assert type(result3) == type(plt.figure())
        plt.close()

        result4 = kosh_operators.KoshSensitivityPlots(self.dataset['inputs'],
                                                      method='mutual_info_score',
                                                      input_names=self.input_names,
                                                      outputs=self.dataset['outputs'],
                                                      output_names=self.output_names,
                                                      n_neighbors=3,
                                                      save_plot=True)[:]
        assert type(result4) == type(plt.figure())

        result5 = kosh_operators.KoshSensitivityPlots(self.dataset['inputs'],
                                                      method='pce_score',
                                                      input_names=self.input_names,
                                                      outputs=self.dataset['outputs'],
                                                      output_names=self.output_names,
                                                      input_ranges=self.ranges,
                                                      degree=1,
                                                      model_degrees=1,
                                                      save_plot=True)[:]
        assert type(result5) == type(plt.figure())
        plt.close()

        result6 = kosh_operators.KoshSensitivityPlots(self.dataset['inputs'],
                                                      method='f_score_rank',
                                                      input_names=self.input_names,
                                                      outputs=self.dataset['outputs'],
                                                      output_names=self.output_names,
                                                      degree=1,
                                                      save_plot=True)[:]
        assert type(result6) == type(plt.figure())
        plt.close()

        result7 = kosh_operators.KoshSensitivityPlots(self.dataset['inputs'],
                                                      method='mutual_info_rank',
                                                      input_names=self.input_names,
                                                      outputs=self.dataset['outputs'],
                                                      output_names=self.output_names,
                                                      n_neighbors=3,
                                                      save_plot=True)[:]
        assert type(result7) == type(plt.figure())
        plt.close()

        result8 = kosh_operators.KoshSensitivityPlots(self.dataset['inputs'],
                                                      method='pce_rank',
                                                      input_names=self.input_names,
                                                      outputs=self.dataset['outputs'],
                                                      output_names=self.output_names,
                                                      input_ranges=self.ranges,
                                                      degree=1,
                                                      model_degrees=1,
                                                      save_plot=True)[:]
        assert type(result8) == type(plt.figure())

        result9 = kosh_operators.KoshSensitivityPlots(self.dataset['inputs'],
                                                      method='f_score_network',
                                                      input_names=self.input_names,
                                                      outputs=self.dataset['outputs'],
                                                      output_names=self.output_names,
                                                      degree=3,
                                                      save_plot=True)[:]
        assert type(result9) == type(plt.figure())
        plt.close()

        result10 = kosh_operators.KoshSensitivityPlots(self.dataset['inputs'],
                                                       method='pce_network',
                                                       input_names=self.input_names,
                                                       outputs=self.dataset['outputs'],
                                                       output_names=self.output_names,
                                                       input_ranges=self.ranges,
                                                       degree=3,
                                                       model_degrees=3,
                                                       save_plot=True)[:]
        assert type(result10) == type(plt.figure())
        plt.close()

        for m in methods:
            plot_name = f"output_A_output_B_{m}_plot.png"
            assert os.path.exists

        for m in methods:
            os.remove(f"output_A_output_B_{m}_plot.png")


if __name__ == '__main__':
    unittest.main()
