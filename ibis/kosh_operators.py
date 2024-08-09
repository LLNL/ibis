import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
from kosh.operators.core import KoshOperator
from ibis import mcmc, sensitivity


class KoshMCMC(KoshOperator):
    """
    Uses a kosh datasets as inputs and outputs and converts them into IBIS input
    and output variables. With other information the user provides such as a surrogate
    model, prior distribution, etc. MCMC sampling chains are
    """

    types = {"numpy": ["ibis/mcmc", ]}

    def __init__(self, *args, **options):
        """
        :param method: Type of MCMC sampling to use. Options are: "default_mcmc"
        and "discrepancy_mcmc".
        :type method: str
        :param inputs: The input datasets. Kosh datasets of one or more arrays.
        The arrays should have features as columns and observation as rows.
        :type inputs: Kosh datasets
        :param input_names:
        :type input_names:
        :param inputs_low: The lower bounds of the features in the input datasets.
        :type inputs_low: list
        :param inputs_high: The upper bounds of the features in the input datasets.
        :type inputs_high: list
        :param proposal_sigmas: The standard deviation of the proposal distribution
        for each feature in the inputs.
        :type proposal_sigmas: list
        :param prior: The prior distributions of each input feature. The default is
        None which will result in using a uniform distribution over the whole range.
        :type prior: list of functions
        :param unscaled_low:
        :type unscaled_low:
        :param unscaled_high:
        :type unscaled_high:
        :param scaling: The type of scaling used on inputs. 'lin' or 'log'
        :type scaling: str
        :param outputs: The output datasets. Kosh datasets of one or more arrays.
        The arrays should have features as columns and observation as rows.
        :type outputs: Kosh datasets
        :param output_names:
        :type output_names:
        :param event: The event the output is associated with
        :type event: str
        param quantity: The physical quantity of the output
        :type quantity: str
        :param surrogate_model:The surrogate model that represents the mapping of input
        values to output values. This model must be have a predict method that takes a
        numpy array and returns a numpy array (sklearn's fit/predict paradigm).
        :type surrogate_model: Trained model with predict method
        :param observed_values: The observed experimental values for each predicted
        output.
        :type observed_values: list of floats
        :param observed_std: The error bound on the observed experimental values for
        each predicted output.
        :type observed_std: list of floats
        :param total_samples: The total number of sample points to return
        :type total_samples: int
        :param burn: The number of burn-in iterations
        :type burn: int
        :param every: The rate at which to save points. Saves every Nth iteration.
        :type every: int
        :param start: The value at which to start the chains
        :type start: dict of str, float
        :param n_chains: The number of chains to run in parallel
        :type n_chains: int
        :param prior_only: Whether to run the chain on just the prior distributions.
        :type prior_only: bool
        :param seed: The random seed for the chains
        :type seed: int
        :param flattened: Whether to flatten the arrays
        :type flattened: bool
        :param scaled: Whether the inputs were scaled
        :type scaled: bool
        """

        super(KoshMCMC, self).__init__(*args, **options)
        self.options = options

    def operate(self, *inputs, format=None):

        # Read in input kosh datasets into one numpy array
        X = inputs[0][:]
        for input_ in inputs[1:]:
            X = np.append(X, input_[:], axis=0)
        Ndim = X.shape[1]

        method = self.options.get("method", "default_mcmc")
        input_names = self.options.get("input_names")
        inputs_low = self.options.get("inputs_low")
        inputs_high = self.options.get("inputs_high")
        proposal_sigmas = self.options.get("proposal_sigmas")
        prior = self.options.get("prior", [sts.uniform.pdf]*Ndim)
        unscaled_low = self.options.get("unscaled_low", None)
        unscaled_high = self.options.get("unscaled_high", None)
        scaling = self.options.get("scaling", None)
        output_names = self.options.get("output_names")
        surrogate_model = self.options.get("surrogate_model")
        event = self.options.get("event", "NoneEvent")
        quantity = self.options.get("quantity", "NoneQuantity")
        observed_values = self.options.get("observed_values")
        observed_std = self.options.get("observed_std")
        total_samples = self.options.get("total_samples")
        burn = self.options.get("burn")
        every = self.options.get("every")
        start = self.options.get("start", None)
        n_chains = self.options.get("n_chains", -1)
        prior_only = self.options.get("prior_only", False)
        seed = self.options.get("seed", None)
        flattened = self.options.get("flattened", False)
        scaled = self.options.get("scaled", True)

        if method == "default_mcmc":
            mcmc_obj = mcmc.DefaultMCMC()
        elif method == "discrepancy_mcmc":
            mcmc_obj = mcmc.DiscrepancyMCMC()
        else:
            msg = f"The MCMC method entered was '{method}'. Please choose "
            msg += "from 'default_mcmc', or 'discrepancy_mcmc'."
            raise ValueError(msg)

        for i, name in enumerate(input_names):
            if (isinstance(unscaled_low, list)) and (isinstance(unscaled_high, list)):
                unscaled_low = unscaled_low[i]
                unscaled_high = unscaled_high[i]
            mcmc_obj.add_input(name=name,
                               low=inputs_low[i],
                               high=inputs_high[i],
                               proposal_sigma=proposal_sigmas[i],
                               prior=prior[i],
                               unscaled_low=unscaled_low,
                               unscaled_high=unscaled_high,
                               scaling=scaling)
        for i, name in enumerate(output_names):
            mcmc_obj.add_output(event=event,
                                quantity=quantity,
                                surrogate_model=surrogate_model[name],
                                observed_value=observed_values[i],
                                observed_std=observed_std[i],
                                inputs=input_names)
        mcmc_obj.run_chain(total=total_samples,
                           burn=burn,
                           every=every,
                           start=start,
                           n_chains=n_chains,
                           prior_only=prior_only,
                           seed=seed)

        return mcmc_obj


class KoshOneAtATimeEffects(KoshOperator):
    """
    Description
    """

    types = {"numpy": ["numpy", ]}

    def __init__(self, *args, **options):
        """
                :param method: The sampling method that was used. Options are 'OAT'
        or 'MOAT'.
        :type method: string
        :param inputs: The input datasets. Kosh datasets of one or more arrays.
        The arrays should have features as columns and observation as rows.
        :type inputs: Kosh datasets
        :param input_names: The names of the inputs for plots
        :type inputs_names: list of str
        :param outputs: The output dataset. Rows correspond to rows in the feature data.
        The arrays should have features as columns and observation as rows.
        :type outputs: Kosh datasets
        :param output_names: The names of the output variables.
        :type output_names: str
        """
        super(KoshOneAtATimeEffects, self).__init__(*args, **options)
        self.options = options

    def operate(self, *inputs, format=None):

        # Read in input kosh datasets into one numpy array
        X = inputs[0][:]
        for input_ in inputs[1:]:
            X = np.append(X, input_[:], axis=0)

        method = self.options.get("method")
        outputs = self.options.get("outputs")

        # Read in output dataset
        Y = np.array(outputs[:])

        if method == "OAT":
            effects = sensitivity.one_at_a_time_effects(X, Y)

        elif method == "MOAT":
            effects = sensitivity.morris_effects(X, Y)

        else:
            msg = "Method must be 'OAT' or 'MOAT'. Was given "
            msg += f"{method}"
            raise ValueError(msg)

        return effects


class KoshSensitivityPlots(KoshOperator):
    """
    Description
    """

    types = {"numpy": ["matplotlib/fig", ]}

    def __init__(self, *args, **options):
        """
        :param inputs: The input datasets. Kosh datasets of one or more arrays.
        The arrays should have features as columns and observation as rows.
        :type inputs: Kosh datasets
        :param input_names: The names of the inputs for plots
        :type inputs_names: list of str
        :param outputs: The output dataset. Rows correspond to rows in the feature data.
        The arrays should have features as columns and observation as rows.
        :type outputs: Kosh datasets
        :param output_names: The names of the output variables.
        :type output_names: list of str
        :param method: Plot type options are 'lasso', 'sensitivity', 'f_score',
        'mutual_info_score', 'pce_score', 'f_score_rank', 'mutual_info_rank',
        'pce_rank', 'f_score_network', or 'pce_network'.
        :type method: string
        :param degree: Maximum degree of interaction for an f_score plot
        :type degree: int
        :param model_degrees: Maximum degree of interaction for PCE model
        :type model_degrees: int
        :param surrogate_model: The surrogate model which has been fit to data
        :type surrogate_model: model with fit/predict functions
        :param input_ranges: Array-like of feature ranges. Each row is a length
        2 array of the lower and upper bounds.
        :type input_ranges: list of arrays
        :param num_plot_points: Number of points to plot on each dimension sweep
        :type num_plot_points: int
        :param num_seed_points: Number of points to use as default points
        :type num_seed_points: int
        :param seed: The random seed for the chains
        :type seed: int
        :param interaction_only: Whether to only include lowest powers of interaction or
        include higher powers for the f_score_plot.
        :type interaction_only: bool
        :param use_p_value: Whether to use p-values or raw F-score in the f_score_plot
        :type use_p_value: bool
        :param n_neighbors: How many neighboring bins to consider when estimating mutual information
        :type n_neighbors: int
        :param max_size: Maximum size of elements in plot. Measured in points
        :type max_size: float
        :param label_size: Font size of labels. Measured in points
        :type label_size: int
        :param alpha: Opacity of elements in plot.
        :type alpha: float
        :param save_plot: Whether to save the plot
        :type save_plot: bool
        """
        super(KoshSensitivityPlots, self).__init__(*args, **options)
        self.options = options

    def operate(self, *inputs, format=None):
        import string

        # Read in input kosh datasets into one numpy array
        X = inputs[0][:]
        for input_ in inputs[1:]:
            X = np.append(X, input_[:], axis=0)
        Ndim = X.shape[1]

        methods = ["lasso", "sensitivity", "f_score", "mutual_info_score", "pce_score",
                   "f_score_rank", "mutual_info_rank", "pce_rank", "f_score_network",
                   "pce_network"]

        method = self.options.get("method")
        input_names = self.options.get("input_names",
                                       list(string.ascii_lowercase)[:Ndim])
        outputs = self.options.get("outputs")
        output_names = self.options.get("output_names", "response")
        surrogate_model = self.options.get("surrogate_model")
        input_ranges = self.options.get("input_ranges")
        num_plot_points = self.options.get("num_plot_points", 100)
        num_seed_points = self.options.get("num_seed_points", 5)
        seed = self.options.get("seed", 2024)
        degree = self.options.get("degree", 1)
        model_degrees = self.options.get("model_degrees", 1)
        interaction_only = self.options.get("interaction_only", True)
        use_p_value = self.options.get("use_p_value", False)
        n_neighbors = self.options.get("n_neighbors", 3)
        max_size = self.options.get("max_size", 10.0)
        label_size = self.options.get("label_size", 10)
        alpha = self.options.get("alpha", 0.5)
        save_plot = self.options.get("save_plot", True)

        # Read in output dataset
        Y = np.array(outputs[:])

        if method == "lasso":
            fig, ax = plt.subplots(1, len(output_names))
            sensitivity.lasso_path_plot(ax, X, Y, input_names, output_names,
                                        degree=degree, method='lasso')
        elif method == "sensitivity":
            fig, ax = plt.subplots(len(input_names), len(output_names))
            sensitivity.sensitivity_plot(ax, surrogate_model, input_names,
                                         output_names, input_ranges,
                                         num_plot_points=num_plot_points,
                                         num_seed_points=num_seed_points, seed=seed)
        elif method == "f_score":
            fig, ax = plt.subplots(1, len(output_names))
            sensitivity.f_score_plot(ax, X, Y, input_names, output_names,
                                     degree=degree, interaction_only=interaction_only,
                                     use_p_value=use_p_value)
        elif method == "mutual_info_score":
            fig, ax = plt.subplots(1, len(output_names))
            sensitivity.mutual_info_score_plot(ax, X, Y, input_names, output_names,
                                               n_neighbors=n_neighbors)
        elif method == "pce_score":
            fig, ax = plt.subplots(1, len(output_names))
            sensitivity.pce_score_plot(ax, X, Y, input_names, output_names,
                                       input_ranges, degree=degree,
                                       model_degrees=model_degrees)
        elif method == "f_score_rank":
            fig, ax = plt.subplots(1, 1)
            sensitivity.f_score_rank_plot(ax, X, Y, input_names, output_names,
                                          degree=degree, interaction_only=interaction_only,
                                          use_p_value=use_p_value)
        elif method == "mutual_info_rank":
            fig, ax = plt.subplots(1, 1)
            sensitivity.mutual_info_rank_plot(ax, X, Y, input_names, output_names,
                                              n_neighbors=n_neighbors)
        elif method == "pce_rank":
            fig, ax = plt.subplots(1, 1)
            sensitivity.pce_rank_plot(ax, X, Y, input_names, output_names, input_ranges,
                                      degree=degree, model_degrees=model_degrees)
        elif method == "f_score_network":
            fig, ax = plt.subplots((degree-1), len(output_names))
            sensitivity.f_score_network_plot(ax, X, Y, input_names, output_names,
                                             degree=degree, max_size=max_size,
                                             label_size=label_size, alpha=alpha)
        elif method == "pce_network":
            fig, ax = plt.subplots((degree-1), len(output_names))
            sensitivity.pce_network_plot(ax, X, Y, input_names, output_names,
                                         input_ranges, degree=degree,
                                         model_degrees=model_degrees, max_size=max_size,
                                         label_size=label_size, alpha=alpha)
        else:
            msg = f"Method should be one of the following: {methods}."
            msg += f"Was given {method}."
            raise ValueError(msg)

        if save_plot:
            names = "_".join(output_names)
            fileName = f"{names}_{method}_plot.png"
            fig.savefig(fileName)

        return fig
