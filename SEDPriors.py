import bilby


class SEDPriors:
    """
    Class for storing prior information for the pre-defined models
    """

    def __init__(self):
        """
        initialise default values for the parameters in each of the models
        these are only used for initialising the gaussian process object in a
        bilby.core.likelihood.GeorgeLikelihood() object, and from reading the
        George and Bilby docs (and coded examples) they do not have an effect on
        the actual fit
        """

        # a = 100, b = 1, c = 1, d = -1, S_norm = 100, alpha = -1
        self.retrig_gp_defaults = [100, 1, 1, -1, 100, -1]
        # C = 1, alpha = -0.5
        self.linear_gp_defaults = [1, -0.5]
        # a = 1, b = 1, c = -1
        self.orienti_gp_defaults = [1, 1, -1]
        # a = 100, b = 1, c = 1, d = -1
        self.snellen_gp_defaults = [100, 1, 1, -1]
        return

    def __add_gp_prior__(self, prior_obj: bilby.core.prior.PriorDict):
        """
        Adding the gaussian process parameters to the prior so that covariance from GLEAM
        can be fit jointly with any given model
        """

        # Add the noise to the
        # log constant is the factor out the front but the LOG of it! (update 24/04/2023)
        # log minimum =-10 corresponds to real min = 4.5e-5, log max 0.695 corresponds to real max = 2
        # prior_obj["kernel:k1:log_constant"] = bilby.prior.Uniform(minimum=-10, maximum=30, name="log_A", latex_label=r"$\ln A$")
        prior_obj["kernel:k1:log_constant"] = bilby.prior.Uniform(
            minimum=-15, maximum=-0.5, name="log_A", latex_label=r"$\ln A$"
        )
        # M0 is the scale factor - we don't want this to go beyond about 200MHz which is the width of the GLEAM band
        # update 24/04/2023 - this 'scale factor' is in fact the distance over which the kernel operates,
        # this means it MUST be >= 0
        # and the gleam sub-band measurements are separated by ~7 MHz with four sub-band measurements in a band
        # so taking the log of this we have min = 1.6 (for actual min ~5MHz) and max = 3.4 (for actual max ~30MHz)
        # I think in fact because this is logged we want a Log Uniform distribution (but check this!!)
        # prior_obj["kernel:k2:metric:log_M_0_0"] = bilby.prior.Uniform(minimum=-10, maximum=5.5, name="log_M_0_0", latex_label=r"$\ln M_{00}$")
        prior_obj["kernel:k2:metric:log_M_0_0"] = bilby.prior.LogUniform(
            minimum=1.6, maximum=3.4, name="log_M_0_0", latex_label=r"$\ln M_{00}$"
        )

        # white noise prior for variance
        # prior_obj["white_noise:value"] = bilby.prior.Uniform(minimum=0, maximum=0.5, name="white noise", latex_label=r"$\sigma$")

        return prior_obj

    ############################################################################
    def linear_prior(self, prefix=""):
        """
        A prior for the power law model
        """

        self.linear_prior_dict = bilby.core.prior.PriorDict()

        # prior for C some scale factor no idea so log uniform between 0.001 and 1e4 = 1,000.
        self.linear_prior_dict[prefix + "C"] = bilby.prior.LogUniform(
            minimum=1e-4, maximum=1e4
        )

        # prior for alpha (spectral index) - uniform between -3 and 3
        self.linear_prior_dict[prefix + "alpha"] = bilby.prior.Uniform(
            minimum=-4, maximum=4
        )

        return self.linear_prior_dict

    def linear_gp_prior(self):
        """
        A prior for the power law model with the addition of the gaussian process to model
        covariance
        """
        # make retrig prior with mean prefix
        self.linear_gp_prior_dict = self.linear_prior(prefix="mean:")

        # add priors for noise
        self.linear_gp_prior_dict = self.__add_gp_prior__(self.linear_gp_prior_dict)

        return self.linear_gp_prior_dict

    ############################################################################
    def orienti_prior(self, prefix=""):
        """
        A prior for the parabolic model
        """

        self.orienti_prior_dict = bilby.core.prior.PriorDict()

        # we don't really know anything about a - make it uniform
        self.orienti_prior_dict[prefix + "a"] = bilby.prior.Uniform(
            -100, -1, "a"
        )  # -100 -1e-2

        # we don't know anythign about b - so make it uniform? It must be positive
        self.orienti_prior_dict[prefix + "b"] = bilby.prior.Uniform(1, 50, "b")

        # truncated gaussian for curvature parameter 'c', because we are only interested
        # in fits with a negative curvature!!
        self.orienti_prior_dict[prefix + "c"] = bilby.prior.TruncatedGaussian(
            mu=-1, sigma=5, name="c", maximum=0, minimum=-1e12
        )  # -10, 1

        return self.orienti_prior_dict

    def orienti_gp_prior(self):
        """
        A prior for the parabolic model with the addition of the gaussian process to model
        covariance
        """

        # make retrig prior with mean prefix
        self.orienti_gp_prior_dict = self.orienti_prior(prefix="mean:")

        # add priors for noise
        self.orienti_gp_prior_dict = self.__add_gp_prior__(self.orienti_gp_prior_dict)

        return self.orienti_gp_prior_dict

    ############################################################################
    def snellen_prior(self, prefix=""):
        """
        A prior for the simple absorbed power law model
        """

        self.snellen_prior_dict = bilby.core.prior.PriorDict()

        # prior for a (peak freq) - from ~50MHz to 500 GHz (50-500e3) so we want LogUniform
        self.snellen_prior_dict[prefix + "a"] = bilby.prior.LogUniform(
            minimum=50, maximum=50e3
        )

        # prior for b (peak flux) - from ~50mJy to 10Jy (50e-3 - 10) so LogUniform again
        self.snellen_prior_dict[prefix + "b"] = bilby.prior.LogUniform(
            minimum=30e-3, maximum=15
        )

        # prior for c (alpha_thick) - try uniform [0,3] #adjusted to 4 to allow for extreme sources as in fig. 24 of Callingham+2017
        self.snellen_prior_dict[prefix + "c"] = bilby.prior.Uniform(
            minimum=0, maximum=4
        )

        # prior for d (alpha_thin) - try uniform [-3, 0] #adjusted to be even with alpha_thick
        self.snellen_prior_dict[prefix + "d"] = bilby.prior.Uniform(
            minimum=-4, maximum=0
        )

        return self.snellen_prior_dict

    def snellen_gp_prior(self):
        """
        A prior for the simple absorbed power law model with the addition of the gaussian process to model
        covariance
        """

        # make retrig prior with mean prefix
        self.snellen_gp_prior_dict = self.snellen_prior(prefix="mean:")

        # add priors for noise
        self.snellen_gp_prior_dict = self.__add_gp_prior__(self.snellen_gp_prior_dict)

        return self.snellen_gp_prior_dict

    ############################################################################
    def retrig_prior(self, prefix=""):
        """
        A prior for the retriggered model
        """

        # bilby priors
        self.retrig_prior_dict = bilby.core.prior.PriorDict()

        # prior for a (peak freq) - from ~50MHz to 500 GHz (50-500e3) so we want LogUniform
        self.retrig_prior_dict[prefix + "a"] = bilby.prior.LogUniform(
            minimum=50, maximum=50e3
        )

        # prior for b (peak flux) - from ~50mJy to 10Jy (50e-3 - 10) so LogUniform again
        self.retrig_prior_dict[prefix + "b"] = bilby.prior.LogUniform(
            minimum=30e-3, maximum=15
        )

        # prior for c (alpha_thick) - try uniform [0,3] #adjusted to 4 to allow for extreme sources as in fig. 24 of Callingham+2017
        self.retrig_prior_dict[prefix + "c"] = bilby.prior.Uniform(minimum=0, maximum=4)

        # prior for d (alpha_thin) - try uniform [-3, 0] #adjusted to be even with alpha_thick
        self.retrig_prior_dict[prefix + "d"] = bilby.prior.Uniform(
            minimum=-4, maximum=0
        )

        # prior for Snorm (norm factor on power law)
        self.retrig_prior_dict[prefix + "S_norm"] = bilby.prior.LogUniform(
            minimum=1, maximum=1e5
        )

        # prior for alpha (spectral index on power law)
        self.retrig_prior_dict[prefix + "alpha"] = bilby.prior.Uniform(
            minimum=-4, maximum=0
        )

        return self.retrig_prior_dict

    def retrig_gp_prior(self):
        """
        A prior for the retriggered model with the addition of the gaussian process to model
        covariance
        """

        # make retrig prior with mean prefix
        self.retrig_gp_prior_dict = self.retrig_prior(prefix="mean:")
        # add priors for noise
        self.retrig_gp_prior_dict = self.__add_gp_prior__(self.retrig_gp_prior_dict)
        return self.retrig_gp_prior_dict

    ############################################################################
    # A custom prior of your choosing!
    def custom_prior(self, name="custom_prior", prior_dict: dict = None):
        """
        A custom prior for a model of your choosing! This is really a very thin wrapper around
        a bilby.core.prior.PriorDict object. You must supply a name for your prior,
        and a dictionary of prior bounds and parameters. See
        """

        self.name = bilby.core.prior.PriorDict(prior_dict)

        for k, v in prior_dict.items():
            self.name[k] = v
