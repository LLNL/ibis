from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import pytest

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import scipy.stats as sts
from sklearn.gaussian_process import GaussianProcessRegressor as gpr

from ibis import mcmc
from ibis import filter
from ibis import likelihoods



X = np.array([[0.46123442, 0.91202244, 0.92876321], [0.06385122, 0.08942266, 0.44186744],
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
Y = np.stack([X.sum(axis=1) ** 2, np.sin(X.sum(axis=1))]).T
input_names = ['input_A', 'input_B', 'input_C']
output_names = ['output_A', 'output_B']
ranges = [[0, 1]] * 3


@pytest.mark.skipif(sys.version_info[0] < 3, reason="Not supported for Python 2")
def test_default_mcmc():
    surrogate_model = {name: gpr().fit(X , Y[:, i]) for i, name in enumerate(output_names)}
    default_mcmc = mcmc.DefaultMCMC()
    for name in input_names:
        default_mcmc.add_input(name, 0.0, 1.0, .2, sts.beta(2, 2).pdf)
    for name in output_names:
        default_mcmc.add_output(name, 'x', surrogate_model[name], .5, .1, input_names)
    default_mcmc.run_chain(total=10, burn=20, every=2, start={name: .5 for name in input_names}, prior_only=True,
                seed=20200221)
    prior_chain_actual = default_mcmc.get_chains(flattened=True)
    prior_diag_actual = default_mcmc.diagnostics_string()
    default_mcmc.run_chain(total=10, burn=20, every=2, start={name: .5 for name in input_names}, prior_only=False,
                seed=20200221)
    posterior_chain_actual = default_mcmc.get_chains(flattened=True)
    post_diag_actual = default_mcmc.diagnostics_string()

    prior_chain_expected = {'input_A': np.array([0.55264703, 0.79214727, 0.89930301, 0.58756906, 0.47650246,
                                                0.14766779, 0.14253315, 0.14253315, 0.3209054, 0.3743087]),
                            'input_B': np.array([0.15907469, 0.20723733, 0.19921041, 0.18157291, 0.17415209,
                                                0.34563449, 0.56326948, 0.56326948, 0.64488806, 0.67500791]),
                            'input_C': np.array([0.56756714, 0.38173197, 0.36747113, 0.20054098, 0.48191384,
                                                0.95840318, 0.68511602, 0.68511602, 0.51473225, 0.39249496])}
    prior_diag_expected = 'acceptance_rate:0.560975609756\n' \
                          'input_A:\n' \
                          '  r_hat:2.2617755855702\n' \
                          '  n_eff:1.3386894239721\n' \
                          '  var_hat:0.1126901165922\n' \
                          '  mean:0.4436117031237\n' \
                          '  std:0.2552577058995\n' \
                          '  mode:0.1463169996242\n' \
                          'input_B:\n' \
                          '  r_hat:3.0080989395777\n' \
                          '  n_eff:1.2969916848321\n' \
                          '  var_hat:0.0767884318154\n' \
                          '  mean:0.3713316843256\n' \
                          '  std:0.2044228036195\n' \
                          '  mode:0.564082263686\n' \
                          'input_C:\n' \
                          '  r_hat:1.3221135286268\n' \
                          '  n_eff:2.7707369877955\n' \
                          '  var_hat:0.0563963666147\n' \
                          '  mean:0.5235087507818\n' \
                          '  std:0.2027403441317\n' \
                          '  mode:0.6817834793133'

    posterior_chain_expected = {'input_A': np.array([0.124974, 0.124974, 0.124974, 0.178624, 0.230599, 0.134530,
                                                    0.222407, 0.222407, 0.400779, 0.400779]),
                                'input_B': np.array([0.067496, 0.067496, 0.067496, 0.362035, 0.081282, 0.080056,
                                                    0.173547, 0.173547, 0.255166, 0.255166]),
                                'input_C': np.array([0.737128, 0.737128, 0.737128, 0.400647, 0.166603, 0.219185,
                                                    0.219786, 0.219786, 0.049403, 0.049403])}
    post_diag_expected = 'acceptance_rate:0.2439024390243\n' \
                         'input_A:\n' \
                         '  r_hat:1.2904145497423\n' \
                         '  n_eff:2.8165865983087\n' \
                         '  var_hat:0.0137083419097\n' \
                         '  mean:0.2165046854049\n' \
                         '  std:0.1007329671144\n' \
                         '  mode:0.1263527688185\n' \
                         'input_B:\n' \
                         '  r_hat:0.9761705708122\n' \
                         '  n_eff:8.6714533971154\n' \
                         '  var_hat:0.0106036355557\n' \
                         '  mean:0.158328713356\n' \
                         '  std:0.098756654963\n' \
                         '  mode:0.0689688225228\n' \
                         'input_C:\n' \
                         '  r_hat:1.7074742053564\n' \
                         '  n_eff:1.8623040191719\n' \
                         '  var_hat:0.1125886842263\n' \
                         '  mean:0.3536195770393\n' \
                         '  std:0.2678458948747\n' \
                         '  mode:0.2178953175562'

    for name in input_names:
        np.testing.assert_array_almost_equal(prior_chain_expected[name], prior_chain_actual[name])
        np.testing.assert_array_almost_equal(posterior_chain_expected[name], posterior_chain_actual[name])
    assert prior_diag_expected == prior_diag_actual
    assert post_diag_expected == post_diag_actual

@pytest.mark.skipif(sys.version_info[0] < 3, reason="Not supported for Python 2")
def test_discrepancy_mcmc():
    surrogate_model = {name: gpr().fit(X , Y[:, i]) for i, name in enumerate(output_names)}
    default_mcmc = mcmc.DiscrepancyMCMC()
    for name in input_names:
        default_mcmc.add_input(name, 0.0, 1.0, .2, sts.beta(2, 2).pdf)
    for name in output_names:
        default_mcmc.add_output(name, 'x', surrogate_model[name], .5, .1, input_names)
    start = {name: .5 for name in input_names}
    start['tau_x'] = .5
    default_mcmc.run_chain(total=10, burn=20, every=2, start=start, prior_only=True,
                seed=20200221)
    prior_chain_actual = default_mcmc.get_chains(flattened=True)
    prior_diag_actual = default_mcmc.diagnostics_string()
    default_mcmc.run_chain(total=10, burn=20, every=2, start=start, prior_only=False,
                seed=20200221)
    posterior_chain_actual = default_mcmc.get_chains(flattened=True)
    post_diag_actual = default_mcmc.diagnostics_string()

    prior_chain_expected = {'input_A': np.array([0.69530471, 0.92059523, 0.85572983, 0.97313971, 0.97313971,
                                                0.7949412, 0.63295273, 0.67205631, 0.6684665, 0.32362777]),
                            'input_B': np.array([0.33595659, 0.66685155, 0.62588496, 0.56605749, 0.56605749,
                                                0.49693199, 0.39095822, 0.57771433, 0.42006784, 0.57253288]),
                            'tau_x'  : np.array([0.73228676, 0.70577684, 0.70697333, 0.66850401, 0.66850401,
                                                0.56500213, 0.54226068, 0.42303957, 0.48123769, 0.40131465]),
                            'input_C': np.array([0.12342421, 0.2150585, 0.67789035, 0.31947208, 0.31947208,
                                                0.38979326, 0.27438913, 0.26592002, 0.26126425, 0.63640871])}
    prior_diag_expected = 'acceptance_rate:0.7073170731707\n' \
                          'input_A:\n' \
                          '  r_hat:1.5448227131246\n' \
                          '  n_eff:2.3879516978398\n' \
                          '  var_hat:0.0528873565946\n' \
                          '  mean:0.7509953688518\n' \
                          '  std:0.1879047224949\n' \
                          '  mode:0.6711166559679\n' \
                          'input_B:\n' \
                          '  r_hat:0.9768757538397\n' \
                          '  n_eff:8.7238377821301\n' \
                          '  var_hat:0.0113273256869\n' \
                          '  mean:0.5219013343058\n' \
                          '  std:0.1020374493138\n' \
                          '  mode:0.5659285875725\n' \
                          'input_C:\n' \
                          '  r_hat:0.9038714596531\n' \
                          '  n_eff:10.0\n' \
                          '  var_hat:0.028614131997\n' \
                          '  mean:0.3483092598274\n' \
                          '  std:0.1682757080035\n' \
                          '  mode:0.3202596903315\n' \
                          'tau_x:\n' \
                          '  r_hat:2.9268965738188\n' \
                          '  n_eff:1.2901396638438\n' \
                          '  var_hat:0.0252183619899\n' \
                          '  mean:0.5894899665642\n' \
                          '  std:0.117416711934\n' \
                          '  mode:0.6677472015219'

    posterior_chain_expected = {'input_A': np.array([0.475518, 0.700808, 0.635943, 0.753353, 0.881327, 0.703129,
                                                    0.642585, 0.43424, 0.43424, 0.089401]),
                                'input_B': np.array([0.203893, 0.534788, 0.493821, 0.433994, 0.485375, 0.416249,
                                                    0.047725, 0.142518, 0.142518, 0.294983]),
                                'tau_x'  : np.array([0.79882096, 0.77231104, 0.77350753, 0.75482242, 0.67644306,
                                                    0.57294118, 0.50934768, 0.45869821, 0.45869821, 0.39599791]),
                                'input_C': np.array([0.383707, 0.475342, 0.938174, 0.579755, 0.582993, 0.653314,
                                                    0.282688, 0.281802, 0.281802, 0.656946])}
    post_diag_expected = 'acceptance_rate:0.7317073170731\n' \
                         'input_A:\n' \
                         '  r_hat:1.2050094810229\n' \
                         '  n_eff:2.8286005956407\n' \
                         '  var_hat:0.0582228250659\n' \
                         '  mean:0.575054418064\n' \
                         '  std:0.2124858342015\n' \
                         '  mode:0.4338889334674\n' \
                         'input_B:\n' \
                         '  r_hat:1.4392577876758\n' \
                         '  n_eff:2.0419984351457\n' \
                         '  var_hat:0.0399932401235\n' \
                         '  mean:0.3195866018674\n' \
                         '  std:0.166491214055\n' \
                         '  mode:0.1427025070428\n' \
                         'input_C:\n' \
                         '  r_hat:1.0488932375744\n' \
                         '  n_eff:10.0\n' \
                         '  var_hat:0.0473150129408\n' \
                         '  mean:0.511652368966\n' \
                         '  std:0.2021390543157\n' \
                         '  mode:0.2850838727684\n' \
                         'tau_x:\n' \
                         '  r_hat:3.3351230809745\n' \
                         '  n_eff:1.2640046361606\n' \
                         '  var_hat:0.0457345129967\n' \
                         '  mean:0.6048789944035\n' \
                         '  std:0.1565628703086\n' \
                         '  mode:0.4436509084991'

    for name in input_names:
        np.testing.assert_array_almost_equal(prior_chain_expected[name], prior_chain_actual[name])
        np.testing.assert_array_almost_equal(posterior_chain_expected[name], posterior_chain_actual[name])
    assert prior_diag_expected == prior_diag_actual
    assert post_diag_expected == post_diag_actual

def test_gaussian_filter():
    gaussian_filter = filter.GaussianFilter()
    weights_actual = gaussian_filter.get_weights(X, .5, .1, sigma_cut=2.0)
    weights_expected = np.array(
        [[0.92761499, 0., 0.], [0., 0., 0.84453447], [0.16568042, 0.97153907, 0.79176711], [0., 0., 0.95627753],
        [0., 0., 0.38817915], [0.90010607, 0.71552566, 0.], [0., 0., 0.93414424], [0., 0.4972952, 0.],
        [0., 0., 0.1865514], [0.96431993, 0., 0.], [0., 0.18297516, 0.52661968], [0.75573119, 0., 0.],
        [0.80632919, 0.82058432, 0.29584603], [0.59989117, 0.26478732, 0.], [0., 0.15398885, 0.],
        [0.99957819, 0., 0.], [0., 0., 0.], [0., 0., 0.], [0.25440143, 0., 0.30526786], [0., 0., 0.9498289],
        [0., 0., 0.], [0.42744552, 0., 0.52456885], [0., 0., 0.45013765], [0.26352929, 0.38162616, 0.72095694],
        [0., 0., 0.], [0.8508424, 0.99922518, 0.15497966], [0.53932804, 0.88456046, 0.], [0., 0., 0.],
        [0., 0., 0.], [0.56540155, 0.96004553, 0.], [0., 0., 0.], [0.19031982, 0.86560632, 0.],
        [0.94880633, 0.35307874, 0.18210179], [0., 0., 0.21777917], [0.15171721, 0.81159391, 0.],
        [0.99858444, 0.77486727, 0.66509396], [0., 0., 0.], [0.47474372, 0.91736596, 0.61001629],
        [0., 0.98762611, 0.], [0.7661603, 0., 0.], [0., 0., 0.96483452], [0., 0., 0.], [0., 0.86236965, 0.],
        [0.81117313, 0.50851489, 0.29053367], [0., 0.25920762, 0.23538773], [0.98021968, 0.99863805, 0.],
        [0., 0., 0.28101814], [0., 0., 0.], [0.55268851, 0., 0.13737018], [0., 0., 0.], [0.81572487, 0., 0.],
        [0., 0.18780823, 0.], [0., 0., 0.], [0.35633297, 0., 0.], [0.16775539, 0., 0.], [0.6463111, 0., 0.],
        [0., 0., 0.], [0., 0., 0.40718963], [0., 0.57186362, 0.72856466], [0., 0.58001078, 0.],
        [0., 0.8525946, 0.], [0., 0., 0.26744788], [0.43895, 0., 0.25362746], [0., 0., 0.], [0., 0.99387378, 0.],
        [0., 0.9994835, 0.57162004], [0., 0., 0.], [0., 0., 0.], [0., 0.22923232, 0.],
        [0., 0.99997093, 0.95746125], [0.99959367, 0., 0.1583589], [0., 0., 0.], [0.99839269, 0., 0.],
        [0.71658875, 0.29173424, 0.40028223], [0.48361092, 0.91242659, 0.], [0., 0.77404633, 0.],
        [0., 0., 0.53367698], [0., 0., 0.], [0., 0.69758618, 0.], [0., 0., 0.],
        [0.26699673, 0.56506223, 0.28557927], [0., 0., 0.], [0., 0., 0.70027344], [0., 0.96972921, 0.5326846],
        [0., 0.93365345, 0.58531589], [0., 0., 0.28325335], [0.16347593, 0.9531238, 0.57220407], [0., 0., 0.],
        [0., 0., 0.84406683], [0., 0., 0.], [0.66518472, 0.66608904, 0.], [0., 0.15394551, 0.],
        [0., 0., 0.80950266], [0., 0., 0.], [0., 0.99613121, 0.], [0., 0.40018062, 0.],
        [0.44192305, 0., 0.55919461], [0., 0., 0.], [0., 0.88693232, 0.], [0., 0., 0.]])
    np.testing.assert_array_almost_equal(weights_expected, weights_actual)

def test_log_gaussian_filter():
    log_gaussian_filter = filter.LogGaussianFilter()
    weights_actual = log_gaussian_filter.get_weights(X, .5, .1, sigma_cut=2.0)
    weights_expected = np.array(
        [[-7.51385096e-02, -8.48812455e+00, -9.19189451e+00], [-9.51128791e+00, -8.42868761e+00, -1.68969727e-01],
        [-1.79769450e+00, -2.88737992e-02, -2.33487985e-01], [-6.16836659e+00, -3.84676104e+00, -4.47071081e-02],
        [-2.23998191e+00, -1.16432106e+01, -9.46288314e-01], [-1.05242664e-01, -3.34737811e-01, -2.51358378e+00],
        [-3.03617803e+00, -3.71836295e+00, -6.81244181e-02], [-5.90010921e+00, -6.98571456e-01, -8.61910733e+00],
        [-9.66503875e+00, -7.68421571e+00, -1.67904847e+00], [-3.63321594e-02, -6.43487866e+00, -2.27470522e+00],
        [-6.83991936e+00, -1.69840486e+00, -6.41276653e-01], [-2.80069541e-01, -8.07367646e+00, -5.60773305e+00],
        [-2.15263196e-01, -1.97738613e-01, -1.21791612e+00], [-5.11007019e-01, -1.32882835e+00, -2.45648201e+00],
        [-2.53719641e+00, -1.87087508e+00, -1.20025006e+01], [-4.21896057e-04, -1.12886066e+01, -9.95853827e+00],
        [-3.33988603e+00, -4.59051454e+00, -8.53088853e+00], [-6.76838325e+00, -2.74711667e+00, -9.92335390e+00],
        [-1.36884181e+00, -9.81751370e+00, -1.18656566e+00], [-1.05757491e+01, -9.73092420e+00, -5.14734201e-02],
        [-6.65184982e+00, -4.37181834e+00, -6.33626067e+00], [-8.49928445e-01, -7.94329325e+00, -6.45178592e-01],
        [-6.51282667e+00, -6.04597456e+00, -7.98201848e-01], [-1.33359077e+00, -9.63313779e-01, -3.27175864e-01],
        [-2.05722823e+00, -2.80390571e+00, -4.36626929e+00], [-1.61528367e-01, -7.75124439e-04, -1.86446141e+00],
        [-6.17431280e-01, -1.22664413e-01, -2.57355112e+00], [-5.24922834e+00, -8.00212414e+00, -8.18683874e+00],
        [-3.62973725e+00, -6.77817282e+00, -5.02935170e+00], [-5.70219088e-01, -4.07745699e-02, -2.66453349e+00],
        [-3.86904094e+00, -3.53501314e+00, -3.75211427e+00], [-1.65904934e+00, -1.44325067e-01, -2.25305119e+00],
        [-5.25505830e-02, -1.04106418e+00, -1.70318944e+00], [-9.24943469e+00, -5.17115980e+00, -1.52427371e+00],
        [-1.88573696e+00, -2.08755170e-01, -9.19818469e+00], [-1.41656209e-03, -2.55063532e-01, -4.07826956e-01],
        [-7.59636551e+00, -5.09794061e+00, -4.44458347e+00], [-7.44980150e-01, -8.62487984e-02, -4.94269613e-01],
        [-7.67203206e+00, -1.24510828e-02, -9.73064892e+00], [-2.66363867e-01, -2.15965651e+00, -7.03065714e+00],
        [-8.05069486e+00, -3.63686108e+00, -3.57986719e-02], [-1.23921193e+01, -6.22793760e+00, -9.54315429e+00],
        [-5.61657006e+00, -1.48071269e-01, -5.69641145e+00], [-2.09273769e-01, -6.76260775e-01, -1.23603582e+00],
        [-6.79767767e+00, -1.35012591e+00, -1.44652122e+00], [-1.99785657e-02, -1.36287940e-03, -6.74020909e+00],
        [-8.69629824e+00, -1.23444182e+01, -1.26933606e+00], [-3.19169385e+00, -7.54640465e+00, -1.06393448e+01],
        [-5.92960718e-01, -3.40462744e+00, -1.98507595e+00], [-1.23775220e+01, -9.02484808e+00, -2.94709671e+00],
        [-2.03678149e-01, -4.83175014e+00, -3.56300808e+00], [-9.90186500e+00, -1.67233391e+00, -1.00805093e+01],
        [-9.18178320e+00, -9.21150673e+00, -9.03162273e+00], [-1.03188967e+00, -1.07201215e+01, -4.79618795e+00],
        [-1.78524835e+00, -2.69746463e+00, -3.29278283e+00], [-4.36474315e-01, -5.95940480e+00, -8.99456775e+00],
        [-2.82954216e+00, -6.64601377e+00, -6.51651967e+00], [-4.19136454e+00, -2.42459380e+00, -8.98476291e-01],
        [-1.07642298e+01, -5.58854738e-01, -3.16678902e-01], [-9.60524851e+00, -5.44708597e-01, -1.06229661e+01],
        [-9.03767247e+00, -1.59471112e-01, -6.68863996e+00], [-1.14625654e+01, -8.77212927e+00, -1.31883059e+00],
        [-8.23369773e-01, -3.01425346e+00, -1.37188879e+00], [-2.58691282e+00, -9.74559021e+00, -1.00242404e+01],
        [-5.24181235e+00, -6.14505849e-03, -8.81796942e+00], [-9.72429784e+00, -5.16637655e-04, -5.59280773e-01],
        [-6.61547105e+00, -8.04949833e+00, -1.22785947e+01], [-5.25884459e+00, -9.33161386e+00, -1.18649304e+01],
        [-3.54000176e+00, -1.47301929e+00, -7.25817497e+00], [-2.72018832e+00, -2.90741251e-05, -4.34700304e-02],
        [-4.06412901e-04, -3.33189739e+00, -1.84289128e+00], [-1.10842380e+01, -3.04460952e+00, -8.76164000e+00],
        [-1.60860756e-03, -9.23973927e+00, -7.06073259e+00], [-3.33253174e-01, -1.23191202e+00, -9.15585404e-01],
        [-7.26474570e-01, -9.16476484e-02, -3.79283526e+00], [-1.03217034e+01, -2.56123551e-01, -3.06373327e+00],
        [-1.17426765e+01, -9.27816291e+00, -6.27964521e-01], [-5.07728339e+00, -6.21601996e+00, -3.89708200e+00],
        [-1.13476141e+01, -3.60129211e-01, -7.56535643e+00], [-3.28804829e+00, -4.11815640e+00, -4.75672806e+00],
        [-1.32051888e+00, -5.70819413e-01, -1.25323563e+00], [-1.01737779e+01, -9.34940082e+00, -3.41640207e+00],
        [-4.63485334e+00, -1.10917542e+01, -3.56284397e-01], [-8.51805589e+00, -3.07384111e-02, -6.29825782e-01],
        [-2.52499010e+00, -6.86499458e-02, -5.35603599e-01], [-2.73498336e+00, -5.25532155e+00, -1.26141357e+00],
        [-1.81108950e+00, -4.80104831e-02, -5.58259578e-01], [-7.24139044e+00, -3.01462619e+00, -6.88792307e+00],
        [-4.88063306e+00, -5.69860799e+00, -1.69523602e-01], [-4.04634993e+00, -5.29932016e+00, -6.24354753e+00],
        [-4.07690504e-01, -4.06331923e-01, -6.54115714e+00], [-3.65861367e+00, -1.87115654e+00, -2.36923761e+00],
        [-5.16960662e+00, -1.00837258e+01, -2.11335224e-01], [-8.80763576e+00, -1.05311856e+01, -7.23931424e+00],
        [-4.43851050e+00, -3.87629559e-03, -4.82215456e+00], [-5.11427456e+00, -9.15839284e-01, -6.73057797e+00],
        [-8.16619507e-01, -1.23373521e+01, -5.81257728e-01], [-2.29007949e+00, -5.20400288e+00, -4.35594560e+00],
        [-7.31266890e+00, -1.19986602e-01, -3.42875714e+00], [-9.67376532e+00, -8.92633836e+00, -6.94371602e+00]])
    np.testing.assert_array_almost_equal(weights_expected, weights_actual)

def test_student_t_filter():
    student_t_filter = filter.StudentTFilter()
    weights_actual = student_t_filter.get_weights(X, .5, .1, sigma_cut=2.0)
    weights_expected = np.array(
        [[0.8693558, 0., 0.], [0., 0., 0.74741798], [0.21760943, 0.94540512, 0.68167442], [0., 0., 0.9179245],
        [0., 0., 0.34571254], [0.82611493, 0.59899048, 0.], [0., 0., 0.88008891], [0., 0.41716328, 0.],
        [0., 0., 0.22945795], [0.9322581, 0., 0.], [0., 0.22743763, 0.43810587], [0.6409685, 0., 0.],
        [0.69904338, 0.71660073, 0.2910503], [0.49455641, 0.27339909, 0.], [0., 0.2108926, 0.],
        [0.99915692, 0., 0.], [0., 0., 0.], [0., 0., 0.], [0.26754538, 0., 0.29646044], [0., 0., 0.90666201],
        [0., 0., 0.], [0.37039, 0., 0.43661312], [0., 0., 0.38514812], [0.27268898, 0.34169022, 0.60446638],
        [0., 0., 0.], [0.75582549, 0.99845215, 0.21146465], [0.44745481, 0.80300076, 0.], [0., 0., 0.],
        [0., 0., 0.], [0.46719406, 0.92459969, 0.], [0., 0., 0.], [0.23158341, 0.77600582, 0.],
        [0.90489453, 0.32445112, 0.22694372], [0., 0., 0.24700217], [0.20957885, 0.70546223, 0.],
        [0.99717488, 0.66219593, 0.55076576], [0., 0., 0.], [0.40161283, 0.85288021, 0.50288171],
        [0., 0.97570289, 0.], [0.65243159, 0., 0.], [0., 0., 0.93318634], [0., 0., 0.], [0., 0.77152008, 0.],
        [0.70494641, 0.42507581, 0.28801249], [0., 0.27025188, 0.25686851], [0.9615781, 0.99728165, 0.],
        [0., 0., 0.28259188], [0., 0., 0.], [0.45747298, 0., 0.20120109], [0., 0., 0.], [0.71055212, 0., 0.],
        [0., 0.23016719, 0.], [0., 0., 0.], [0.32639426, 0., 0.], [0.2187946, 0., 0.], [0.53391747, 0., 0.],
        [0., 0., 0.], [0., 0., 0.35753198], [0., 0.4722083, 0.61223573], [0., 0.47860236, 0.],
        [0., 0.75818332, 0.], [0., 0., 0.27490191], [0.37782335, 0., 0.26710989], [0., 0., 0.],
        [0., 0.9878591, 0.], [0., 0.99896779, 0.47201839], [0., 0., 0.], [0., 0., 0.], [0., 0.25341871, 0.],
        [0., 0.99994186, 0.92001393], [0.99918783, 0., 0.21341152], [0., 0., 0.], [0.9967931, 0., 0.],
        [0.60005772, 0.28869827, 0.35321076], [0.40767254, 0.84509759, 0.], [0., 0.66126759, 0.],
        [0., 0., 0.44327635], [0., 0., 0.], [0., 0.58130801, 0.], [0., 0., 0.],
        [0.27464697, 0.46693214, 0.28518699], [0., 0., 0.], [0., 0., 0.58391815], [0., 0.94208369, 0.44254611],
        [0., 0.87927556, 0.48281022], [0., 0., 0.28386292], [0.21634818, 0.91239131, 0.47247387], [0., 0., 0.],
        [0., 0., 0.74679966], [0., 0., 0.], [0.55084855, 0.55167427, 0.], [0., 0.21086756, 0.],
        [0., 0., 0.70290347], [0., 0., 0.], [0., 0.99230705, 0.], [0., 0.35314743, 0.],
        [0.37976044, 0., 0.46242444], [0., 0., 0.], [0., 0.80646904, 0.], [0., 0., 0.]])
    np.testing.assert_array_almost_equal(weights_expected, weights_actual)

def test_tophat_filter():
    tophat_filter = filter.TophatFilter()
    weights_actual = tophat_filter.get_weights(X, .5, .1, sigma_cut=.1)
    weights_expected = np.array(
        [[1., 0., 0.], [0., 0., 1.], [0., 1., 1.], [0., 0., 1.], [0., 0., 0.], [1., 1., 0.], [0., 0., 1.],
        [0., 0., 0.], [0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 0., 0.],
        [0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 1.], [0., 0., 0.],
        [0., 0., 0.], [0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [1., 1., 0.], [0., 1., 0.], [0., 0., 0.],
        [0., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 0.], [0., 1., 0.],
        [1., 1., 1.], [0., 0., 0.], [0., 1., 1.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.], [0., 0., 0.],
        [0., 1., 0.], [1., 0., 0.], [0., 0., 0.], [1., 1., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
        [0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [1., 0., 0.],
        [0., 0., 0.], [0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.],
        [0., 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 1., 1.],
        [1., 0., 0.], [0., 0., 0.], [1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 0., 0.],
        [0., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
        [0., 1., 0.], [0., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [1., 1., 0.],
        [0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
        [0., 1., 0.], [0., 0., 0.]])
    np.testing.assert_array_almost_equal(weights_expected, weights_actual)

def test_log_tophat_filter():
    log_tophat_filter = filter.LogTophatFilter()
    weights_actual = log_tophat_filter.get_weights(X, .5, .1, sigma_cut=1.0)
    with np.errstate(divide = 'ignore'):
        weights_expected = np.log(np.array(
        [[1., 0., 0.], [0., 0., 1.], [0., 1., 1.], [0., 0., 1.], [0., 0., 0.], [1., 1., 0.], [0., 0., 1.],
        [0., 0., 0.], [0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 0., 0.],
        [0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 1.], [0., 0., 0.],
        [0., 0., 0.], [0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [1., 1., 0.], [0., 1., 0.], [0., 0., 0.],
        [0., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 0.], [0., 1., 0.],
        [1., 1., 1.], [0., 0., 0.], [0., 1., 1.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.], [0., 0., 0.],
        [0., 1., 0.], [1., 0., 0.], [0., 0., 0.], [1., 1., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
        [0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [1., 0., 0.],
        [0., 0., 0.], [0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.],
        [0., 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 1., 1.],
        [1., 0., 0.], [0., 0., 0.], [1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 0., 0.],
        [0., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
        [0., 1., 0.], [0., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [1., 1., 0.],
        [0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
        [0., 1., 0.], [0., 0., 0.]]))
    np.testing.assert_array_almost_equal(weights_expected, weights_actual)

def test_gaussian_mixture_filter():
    gaussian_mixture_filter = filter.GaussianMixtureFilter()
    weights_actual = gaussian_mixture_filter.get_weights(X,
                                                        np.array([[.5, .5, .5]]),
                                                        np.array([[.1, .1, .1]]),
                                                        sigma_cut=1.0)
    weights_expected = np.array(
        [4.37792252e-07, 1.22635664e-08, 4.27240660e-02, 1.08430762e-05, 2.83919282e-06, 5.09425641e-02,
        3.99032364e-03, 7.88128246e-10, 2.53143382e-11, 6.61676602e-05, 1.41738133e-07, 8.83500044e-06,
        5.47061014e-01, 5.28981479e-02, 7.94884356e-11, 1.49700648e-08, 2.23728835e-08, 8.56080092e-12,
        1.98303902e-06, 1.99272849e-14, 3.77411570e-10, 6.67191779e-06, 1.23993011e-05, 2.20593771e-01,
        9.51454278e-07, 2.94645440e-01, 1.40026516e-02, 6.73565618e-10, 3.09132051e-11, 2.32719752e-02,
        8.11364818e-09, 6.66522631e-02, 1.40462552e-01, 1.20045446e-12, 2.93462203e-07, 1.00000000e+00,
        1.76981259e-09, 2.82736610e-01, 2.33547465e-07, 5.58038593e-04, 8.22502552e-07, 1.96183627e-10,
        8.25297416e-05, 8.98472412e-02, 9.11615625e-07, 1.80796066e-03, 5.57808773e-11, 2.37708735e-14,
        1.26530773e-02, 8.07574966e-10, 1.00970734e-03, 1.35284188e-16, 8.17640811e-11, 7.40366273e-07,
        4.09178355e-04, 4.55832405e-06, 1.27923513e-07, 5.45009523e-05, 1.48443135e-08, 5.37882463e-14,
        1.62177517e-06, 4.45376060e-09, 1.44418581e-02, 1.54960721e-08, 9.97034709e-10, 7.25834302e-08,
        2.97161644e-11, 2.01716828e-18, 3.25643155e-07, 4.31171542e-02, 5.43098615e-03, 2.29337980e-18,
        1.09119720e-06, 1.47557026e-01, 4.75360382e-03, 5.30756739e-11, 9.52326418e-17, 1.98174927e-11,
        1.44757667e-08, 5.24158139e-09, 3.47596092e-02, 5.98150173e-09, 1.75290783e-06, 5.68044323e-07,
        2.39255184e-02, 4.30244784e-05, 9.32471147e-02, 9.34501905e-08, 6.96951739e-08, 6.83094453e-07,
        2.03020809e-04, 9.35190651e-05, 1.32464042e-09, 2.38578815e-13, 3.84092347e-07, 7.33371743e-06,
        6.53855228e-08, 5.03447613e-06, 6.46201142e-05, 2.61830523e-10])
    np.testing.assert_array_almost_equal(weights_expected, weights_actual)

def test_intersection_likelihood():
    likelihood = likelihoods.Intersection()
    weights = filter.GaussianFilter().get_weights(X, 0.5, .3, sigma_cut=1.0)
    combined_weights_actual = likelihood.combine_weights(weights.T)
    combined_weights_expected = np.array(
        [0., 0., 0.04382461, 0., 0., 0.03968276, 0.02581658, 0., 0., 0., 0., 0., 0.04596488, 0.03418278, 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.04116244, 0.01976324, 0.04398702, 0.03812643, 0., 0., 0.03828826, 0.01595092,
        0.03510617, 0.04038, 0., 0., 0.05117642, 0., 0.04755149, 0., 0., 0., 0., 0., 0.0435261, 0., 0., 0., 0.,
        0.02834216, 0., 0., 0., 0., 0., 0.02322308, 0., 0., 0.02390658, 0., 0., 0., 0., 0.03088455, 0., 0., 0., 0.,
        0., 0., 0.04052886, 0.03100254, 0., 0., 0.04182322, 0.0330084, 0., 0., 0., 0., 0., 0.03884944, 0., 0., 0.,
        0.03891567, 0., 0.04211884, 0., 0., 0., 0., 0.02290656, 0., 0., 0., 0., 0., 0., 0., 0.])
    np.testing.assert_array_almost_equal(combined_weights_expected, combined_weights_actual)

def test_stat_filtering_likelihood():
    likelihood = likelihoods.StatFiltering()
    weights = filter.GaussianFilter().get_weights(X, 0.5, .3, sigma_cut=1.0)
    combined_weights_actual = likelihood.combine_weights(weights.T)
    combined_weights_expected = np.array(
        [0.07467045, 0.07282764, 0.89112033, 0.25198332, 0.26411249, 0.82967503, 0.60122491, 0.06324737, 0.04871249,
        0.30043144, 0.29502251, 0.0707044, 0.92043968, 0.73089219, 0.22343896, 0.07617115, 0.03130023, 0.03665282,
        0.28552158, 0.07514255, 0.02377903, 0.32882904, 0.06159127, 0.84212818, 0.47373663, 0.89486525, 0.8029613,
        0., 0.02897499, 0.8075586, 0.39395411, 0.7484784, 0.83286667, 0.05076233, 0.30672659, 1., 0.02332266,
        0.94363234, 0.07592752, 0.2943172, 0.25802893, 0., 0.07323411, 0.88057544, 0.276471, 0.40006575, 0.0543285,
        0.03256023, 0.63565797, 0.03475183, 0.25325687, 0.04879967, 0., 0.05787497, 0.53473814, 0.06781982,
        0.035857, 0.55882105, 0.35713056, 0.06589284, 0.07301211, 0.05361708, 0.67615362, 0.03825045, 0.07605514,
        0.37281495, 0., 0., 0.20907866, 0.84900612, 0.6884665, 0.03386096, 0.07614708, 0.85247816, 0.73072619,
        0.26545386, 0.06444791, 0.02698374, 0.06921285, 0.15002723, 0.8034145, 0.03066886, 0.06928375, 0.36790945,
        0.81637602, 0.23754028, 0.86240461, 0.03413243, 0.0728169, 0.02593211, 0.35998965, 0.53026133, 0.07201057,
        0., 0.2386372, 0.0596916, 0.33313067, 0.16821533, 0.26039306, 0.])
    np.testing.assert_array_almost_equal(combined_weights_expected, combined_weights_actual)
