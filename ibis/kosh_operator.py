import numpy as np
from kosh.operators.core import KoshOperator
from ibis import mcmc


class KoshMCMC(KoshOperator):
	"""
	"""

	types = {"numpy": ["numpy", ]}

	def __init__(self, *args, **options):
		"""
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
		:param output_names:
		:type output_names:
		:param event: The event the output is associated with
		:type event: str
		param quantity: The physical quantity of the output
		:type quantity: str
		:param observed_values: The observed experimental values for each predicted
		output.
		:type observed_values: list of floats
		:param observed_std: The error bound on the observed experimental values for
		each predicted output.
		:type observed_std: list of floats
		:param default_mcmc: Whether to use the default MCMC.
		:type default_mcmc: bool
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
		:param seed:
		:type seed: int
		:param surrogate_model:
		:type surrogate_model:
		"""

		super(KoshMCMC, self).__init__(*args, **options)
		self.options = options

	def operate(self, *inputs, **kargs):

        # Read in input kosh datasets into one numpy array
        X = inputs[0][:]
        for input_ in inputs[1:]:
            X = np.append(X, input_[:], axis=0)

        Nsamp, Ndim = X.shape

		self.default_mcmc = self.options.get("default_mcmc", 0)
		self.iput_names = self.options.get("input_names")
		self.inputs_low = self.options.get("inputs_low")
		self.inputs_high = self.options.get("inputs_high")
		self.proposal_sigmas = self.options.get("proposal_sigmas")
		self.priors = self.options.get("priors", [None]*Ndim)
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

        if default_mcmc:
        	default_mcmc = mcmc.DefaultMCMC()
        	for i, name in enum(self.input_names):
        		default_mcmc.add_input(name=name,
        							   low=self.inputs_low[i],
        							   high=self.inputs_high[i],
        							   proposal_sigma=self.proposal_sigmas[i],
        							   prior=self.priors[i])
        	for i, name in enum(self.output_names):
        		default_mcmc.add_output(event=name,
        								quantity='x',
        								surrogate_model=self.surrogate_model[name],
        								observed_value=self.observed_values[i],
        								observed_std=self.observed_std[i],
        								inputs=self.input_names)
        	default_mcmc.run_chain(total=self.total_samples,
        						   burn=self.burn,
        						   every=self.every,
        						   start=self.start,
        						   prior_only=self.prior_only,
        						   seed=self.seed)
        	prior_chain_actual = default_mcmc.get_chains(flattened=True)
        	prior_diag_actual = default_mcmc.diagnostics_string()
        	default_mcmc.run_chain()
