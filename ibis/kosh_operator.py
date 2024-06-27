import numpy as np
from kosh.operators.core import KoshOperator
from ibis import mcmc, sensitivity


class KoshMCMC(KoshOperator):
	"""
	Description
	"""

	types = {"numpy": ["numpy", ]}

	def __init__(self, *args, **options):
		"""
		:param method: Type of MCMC sampling to use. Options are: "default_mcmc",
		"mcmc", "discrepancy_mcmc".
		:type method: str
		:param inputs: The input datasets. Kosh datasets of one or more arrays.
		The arrays should have features as columns and observation as rows.
		:type inputs: Kosh datasets
		:param Y: The output datasets. Kosh datasets of one or more arrays.
		The arrays should have features as columns and observation as rows.
		:type Y: Kosh datasets
		:param input_names:
		:type input_names:
		:param inputs_low: The lower bounds of the features in the input datasets.
		:type inputs_low: list
		:param inputs_high: The upper bounds of the features in the input datasets.
		:type inputs_high: list
		:param proposal_sigmas: The standard deviation of the proposal distribution
		for each feature in the inputs.
		:type proposal_sigmas: list
		:param priors: The prior distributions of each input feature. The default is
		None which will result in using a uniform distribution over the whole range.
		:type priors: list of functions
		:param unscaled_low:
		:type unscaled_low:
		:param unscaled_high:
		:type unscaled_high:
		:param scaling: The type of scaling used on inputs. 'lin' or 'log'
		:type scaling: str
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
		:param flattened:
		:type flattened: bool
		"""

		super(KoshMCMC, self).__init__(*args, **options)
		self.options = options

	def operate(self, *inputs, **kargs):

		# Read in input kosh datasets into one numpy array
		X = inputs[0][:]
		for input_ in inputs[1:]:
			X = np.append(X, input_[:], axis=0)

		Nsamp, Ndim = X.shape

		self.method = self.options.get("method", "default_mcmc")
		self.iput_names = self.options.get("input_names")
		self.inputs_low = self.options.get("inputs_low")
		self.inputs_high = self.options.get("inputs_high")
		self.proposal_sigmas = self.options.get("proposal_sigmas")
		self.priors = self.options.get("priors", [None]*Ndim)
		self.unscaled_low = self.options.get("unscaled_low", None)
		self.unscaled_high = self.options.get("unscaled_high", None)
		self.scaling = self.options.get("scaling", None)
		self.output_names = self.options.get("output_names")
		self.surrogate_model = self.options.get("surrogate_model")
		self.event = self.options.get("event", "NoneEvent")
		self.quantity = self.options.get("quantity", "NoneQuantity")
		self.observed_values = self.options.get("observed_values")
		self.observed_std = self.options.get("observed_std")
		self.total_samples = self.options.get("total_samples")
		self.burn = self.options.get("burn")
		self.every = self.options.get("every")
		self.start = self.options.get("start", None)
		self.n_chains = self.options.get("n_chains", -1)
		self.prior_only = self.options.get("prior_only", False)
		self.seed = self.options.get("seed", None)
		self.flattened = self.options.get("flattened", False)

		if method == "mcmc":	
			mcmc_obj = mcmc.MCMC()
        elif method == "default_mcmc":
        	mcmc_obj = mcmc.DefaultMCMC()
        elif method == "discrepancy_mcmc":
        	mcmc_obj = mcmc.DiscrepancyMCMC()
		else:
			msg = f"The MCMC method entered was '{method}'. Please choose "
			msg +=  "from 'default_mcmc', 'mcmc', or 'discrepancy_mcmc'."
			raise ValueError(msg)

		for i, name in enum(self.input_names):
			mcmc_obj.add_input(name=name,
							   low=self.inputs_low[i],
							   high=self.inputs_high[i],
							   proposal_sigma=self.proposal_sigmas[i],
							   prior=self.priors[i])
		for i, name in enum(self.output_names):
			mcmc_obj.add_output(event=name,
								quantity='x',
								surrogate_model=self.surrogate_model[name],
								observed_value=self.observed_values[i],
								observed_std=self.observed_std[i],
								inputs=self.input_names)
		mcmc_obj.run_chain(total=self.total_samples,
						   burn=self.burn,
						   every=self.every,
						   start=self.start,
						   prior_only=self.prior_only,
						   seed=self.seed)
		prior_chain_actual = mcmc_obj.get_chains(flattened=self.flattened,
												 scaled=self.scaled)


class KoshOneAtATimeEffects(KoshOperator):
	"""
	Description
	"""

	types = {"numpy": ["numpy", ]}

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
		:param output_name: The name of the output variable.
		:type output_name: str
		:param return_option: Whether to return 'coefficients' or a 'plot'.
		:type return_option: str
		:param method: The sampling method that was used. Options are 'OAT',
		or 'MOAT'
		:type method: string
		"""
		super(KoshSensitivity, self).__init__(*args, **options)
		self.options = options

	def operate(self, *inputs, **kargs):
		import string

		# Read in input kosh datasets into one numpy array
		X = inputs[0][:]
		for input_ in inputs[1:]:
			X = np.append(X, input_[:], axis=0)
		Nsamp, Ndim = X.shape

		# Read in output dataset
		Y = outputs[:]

		self.return_option = self.options.get("return_option")
		self.method = self.options.get("method")
		self.input_names = self.options.get("input_names",
											list(string.ascii_lowercase(:Ndim)))
		self.output_name = self.options.get("output_name", "response")
		self.degree = self.options.get("degree", "1")
		self.interaction_only = self.options.get("interaction_only", True)
		self.use_p_value = self.options.get("use_p_value", False)

		if method == "OAT":
			effects = sensitivity.one_at_a_time_effects(X, Y)
			coef = np.abs(effects).mean(axis=0)

		elif method == "MOAT":
			effects = sensitivity.morris_effects(X, Y)
			coef = np.abs(effects).mean(axis=0)

		elif method == "FF":
			from sklearn.linear_model import LinearRegression
			FF_model = LinearRegression().fit(FF_samples, FF_response)
			coef = FF_model.coef_

		else:
			msg = f"Method must be one of 'OAT', 'MOAT', or 'FF'. Was given "
			msg += f"{self.method}"
			raise ValueError(msg)

		return = coef


class KoshSensitivityPlots(KoshOperator):
	"""
	Description
	"""

	types = {"numpy": ["numpy", ]}

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
		super(KoshSensitivity, self).__init__(*args, **options)
		self.options = options

	def operate(self, *inputs, **kargs):
		import string

		# Read in input kosh datasets into one numpy array
		X = inputs[0][:]
		for input_ in inputs[1:]:
			X = np.append(X, input_[:], axis=0)
		Nsamp, Ndim = X.shape

		# Read in output dataset
		Y = outputs[:]

		methods = ["lasso", "sensitivity", "f_score", "mutual_info_score", "pce_score",
				   "f_score_rank", "mutual_info_rank", "pce_rank", "f_score_network",
				   "pce_network"]

		self.return_option = self.options.get("return_option")
		self.method = self.options.get("method")
		self.input_names = self.options.get("input_names",
											list(string.ascii_lowercase(:Ndim)))
		self.output_names = self.options.get("output_names", "response")
		self.surrogate_model = self.options.get("surrogate_model")
		self.input_ranges = self.options.get("input_ranges")
		self.num_plot_points = self.options.get("num_plot_points", 100)
		self.num_seed_points = self.options.get("num_seed_points", 5)
		self.seed = self.options.get("seed", 2024)
		self.degree = self.options.get("degree", 1)
		self.model_degrees = self.options.get("model_degrees", 1)
		self.interaction_only = self.options.get("interaction_only", True)
		self.use_p_value = self.options.get("use_p_value", False)
		self.n_neighbors = self.options.get("n_neighbors", 3)
		self.max_size = self.options.get("max_size", 10.0)
		self.label_size = self.options.get("label_size", 10)
		self.alpha = self.options.get("alpha", 0.5)
		self.save_plot = self.options.get("save_plot", True)

		fig, ax = plt.subplots(1, 1)
		if method == "lasso":
			output = sensitivity.lasso_path_plot(ax, X, Y, self.input_names, self.output_names,
										degree=self.degree, method='lasso')
		elif method == "sensitivity":
			output = sensitivity.sensitivity_plot(ax, self.surrogate_model, self.input_names,
										 self.output_names, self.input_ranges,
                     					 num_plot_points=self.num_plot_points,
                     					 num_seed_points=self.num_seed_points, seed=self.seed)
		elif method == "f_score":
			output = sensitivity.f_score_plot([ax], X, Y, self.input_names, self.output_names,
									 degree=self.degree, interaction_only=self.interaction_only,
									 use_p_value=self.use_p_value)
		elif method == "mutual_info_score":
			output = sensitivity.mutual_info_score_plot(ax, X, Y, self.input_names, self.output_names,
											   n_neighbors=self.n_neighbors)
		elif method == "pce_score":
			output = sensitivity.pce_score_plot(ax, X, Y, self.input_names, self.output_names,
									   self.input_ranges, degree=self.degree,
									   model_degrees=self.model_degrees)
		elif method == "f_score_rank":
			output = sensitivity.f_score_rank_plot(ax, X, Y, self.input_names, self.output_names,
										  degree=self.degree, interaction_only=self.interaction_only,
										  use_p_value=self.use_p_value)
		elif method == "mutual_info_rank":
			output = sensitivity.mutual_info_rank_plot(ax, X, Y, self.input_names, self.output_names,
											  n_neighbors=self.n_neighbors)
		elif method == "pce_rank":
			output = sensitivity.pce_rank_plot(ax, X, Y, self.input_names, self.output_names, self.input_ranges,
									  degree=self.degree, model_degrees=self.model_degrees)
		elif method == "f_score_network":
			output = sensitivity.f_score_network_plot(ax, X, Y, self.input_names, self.output_names,
											 degree=self.degree, max_size=self.max_size,
											 label_size=self.label_size, alpha=self.alpha)
		elif method == "pce_network"
			output = sensitivity.pce_network_plot(ax, X, Y, self.input_names, self.output_names,
										 self.input_ranges, degree=self.degree,
										 model_degrees=self.model_degrees, max_size=self.max_size,
										 label_size=self.label_size, alpha=self.alpha)
		else:
			msg = f"Method should be one of the following: {methods}."
			msg += f"Was given {method}."
			raise ValueError(msg)

		if save_plot:
			names = "_".join(output_names)
			fileName = f"{names}_{method}_plot.png"
			output.savefig(fileName)

		return output



	