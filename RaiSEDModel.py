#!/usr/bin/python
import sys
import os
import numpy as np
from types import FunctionType
from inspect import signature

# bilby for mcmc fitting!
import bilby
from helper_functions import get_credible_interval, get_retrig_intervals

####################################################################################################################
#                                        Emily Kerrison (SIFA 2022)                                                #
#               Based on James Allison's SED fitting code from 2014, this function fits                            #
#               a radio SED with the function described in Snellen+1997. The model is for a synchrotron            #
#               emission power law with low-freq absorption. It is fit in linear space. The code then              #
#               outputs the reduced chi-squared, as well as two information criteria, AIC () and BIC ()            #
#               which can be used to distinguish between this and other models to determine which                  #
#               best characterises the SED. It assumes input frequencies in Hz.                                    #
####################################################################################################################


class RaiSEDModel:
    """
    Generic modelling class for my SED code
    """

    # initial function
    def __init__(
        self,
        data,
        model_type: str,
        model_func: FunctionType,
        src_name: str,
        fit_min=1.0e7,
        fit_max=1e11,
        fit_errs=True,
        new_flag=False,
        output_dir="output",
        plot_colour="darkred",
        plot_linestyle="solid",
        savestr_end="",
        use_nestcheck=False,
    ):
        # self.data = data
        self.model_type = model_type
        self.__SED_func__ = model_func
        self.src_name = src_name
        self.fit_min = fit_min
        self.fit_max = fit_max
        self.fit_errs = fit_errs
        self.new_flag = new_flag
        self.output_dir = output_dir
        self.plot_colour = plot_colour
        self.plot_linestyle = plot_linestyle
        self.gp = False
        self.param_intervals = dict()
        self.md = len(signature(self.__SED_func__).parameters) - 1
        self.use_nestcheck=use_nestcheck

        # save suffix
        self.savestr_end = savestr_end

        # get freq, flux and ferr
        self.freq = np.array(data["Frequency (Hz)"][:])
        truth = (self.freq <= self.fit_max) & (self.freq >= self.fit_min)
        self.freq = self.freq[truth].astype("float")
        self.freq /= 1e6  # (to make it MHz for fitting)
        self.flux = np.array(data["Flux Density (Jy)"][truth]).astype("float")
        self.ref = np.array(data["Refcode"][truth])

        # Define data for fitting
        if self.fit_errs:
            self.ferr = np.array(data["Uncertainty"][truth])
        else:
            self.ferr = 1.0

        # set savestr end
        if self.new_flag:
            self.new_yr = new_yr
            self.savestr_end += "new_only"
            age_truth = self.ref.apply(lambda x: int(x[0:4]) > self.new_yr)
            self.freq = self.freq[age_truth]
            self.flux = self.flux[age_truth]

        return

    # setting the prior to a bilby prior
    def set_prior(self, prior: bilby.core.prior.dict.PriorDict):
        self.prior = prior
        self.priorkeys = list(self.prior.keys())
        return

    # setting the likelihood to a bilby likelihood object using the appropriate SED function
    def set_likelihood(self):
        self.likelihood = bilby.likelihood.GaussianLikelihood(
            x=self.freq, y=self.flux, func=self.__SED_func__, sigma=self.ferr
        )
        # self.likelihood = likelihood
        return

    # creating the sampler
    def set_sampler(
        self,
        sampler_type="dynamic_dynesty",
        nlive=512,
        maxbatch=1,
        n_effective=None,
        use_stop=False,
        npool=None,
        check_point=True,
        overwrite=False,
    ):  # mabatch 1
        self.sampler_label = "{}_{}_{}".format(
            self.src_name, self.model_type, self.savestr_end
        )  # no suffix to label
        self.sampler_type = sampler_type
        self.nlive = nlive
        self.npool = npool
        self.check_point = check_point
        self.n_effective = n_effective
        self.overwrite = overwrite

        # if dynamic dynesty parse in other parameters
        if self.sampler_type == "dynamic_dynesty":
            self.maxbatch = maxbatch
            self.use_stop = use_stop
        return

    def setup_sampler(self, prior: bilby.core.prior.dict.PriorDict, **kwargs):
        """Wrapper function for calling set_prior, set_likelihood and set_sampler"""
        self.set_prior(prior)
        self.set_likelihood()
        self.set_sampler(**kwargs)
        return

    def run_sampler(self):
        if self.sampler_type == "dynesty":
            self.result = bilby.run_sampler(
                likelihood=self.likelihood,
                priors=self.prior,
                sampler=self.sampler_type,
                nlive=self.nlive,
                label=self.sampler_label,
                outdir=self.output_dir,
                npool=self.npool,
                check_point=self.check_point,
                clean=self.overwrite,
                n_effective=self.n_effective,
            )

        elif self.sampler_type == "dynamic_dynesty":
            # print('USING DYNAMIC DYNESTY FROM RAISED')
            self.result = bilby.run_sampler(
                likelihood=self.likelihood,
                priors=self.prior,
                sampler=self.sampler_type,
                nlive=self.nlive,
                maxbatch=self.maxbatch,
                use_stop=self.use_stop,
                label=self.sampler_label,
                outdir=self.output_dir,
                npool=self.npool,
                check_point=self.check_point,
                nlive_init=self.nlive,
                nestcheck=self.use_nestcheck,
                clean=self.overwrite,
                n_effective=self.n_effective,
                sample="rwalk",
            )

        else:
            raise Exception("I do not yet understand the sampler you want")

        # make corner plot
        # print('posterior shape!')
        # print(self.result.posterior[self.result.search_parameter_keys].shape)
        # print(self.result.posterior[self.result.search_parameter_keys])
        # try:
        self.result.plot_corner(dpi=150)
        # except:
        #    pass
        # return self.result

    def get_bic(self):
        """returns the Bayesian Information Criterion as defined by Schwarz (1978)"""
        # Define chi-squared calculation
        self.chi_squared = lambda params: np.sum(
            np.power((self.__SED_func__(self.freq, *params) - self.flux) / self.ferr, 2)
        )

        # Calculate best fit chisquared
        self.fit_params = [
            self.result.get_one_dimensional_median_and_error_bar(x).median
            for x in self.priorkeys
        ]
        self.fit_chisq = self.chi_squared(self.fit_params)

        # calculate information criteria
        self.fit_bic = self.fit_chisq + len(self.fit_params) * np.log(len(self.freq))

        return self.fit_bic

    def get_log10z(self):
        """returns the log10 evidence and associated error"""
        try:
            self.log10z = self.result.log_10_evidence
            self.log10z_err = self.result.log_10_evidence_err
            self.log10z_noise = self.result.log_10_noise_evidence
        except AttributeError as e:
            raise AttributeError(
                "{} has no result object, have you called run_sampler() yet?".format(
                    self.model_type
                )
            )
        return self.log10z, self.log10z_err, self.log10z_noise

    def get_lnz(self):
        """returns the natural logarithm of the evidence and associated error"""
        try:
            self.lnz = self.result.log_evidence
            self.lnz_err = self.result.log_evidence_err
        except AttributeError as e:
            raise AttributeError(
                "{} has no result object, have you called run_sampler() yet?".format(
                    self.model_type
                )
            )
        return self.lnz, self.lnz_err

    def get_bmd(self):
        """returns the Bayesian Model Dimensionality"""
        try:
            self.bmz = self.result.bayesian_model_dimensionality
        except AttributeError as e:
            raise AttributeError(
                "{} has no result object, have you called run_sampler() yet?".format(
                    self.model_type
                )
            )
        return self.bmz

    def get_md(self):
        """returns the Model Dimensionality (number of parameters fit)"""
        return self.md

    def get_best_fit_params(self):
        if not hasattr(self, "result"):
            raise AttributeError(
                "Cannot get best fit before running the sampler! Try calling run_sampler() first."
            )

        if not hasattr(self, "fit_params"):
            self.fit_params = [
                self.result.get_one_dimensional_median_and_error_bar(x).median
                for x in self.priorkeys
            ]
        return self.fit_params

    def get_best_fit_func(self):
        if not hasattr(self, "fit_params"):
            self.fit_params = [
                self.result.get_one_dimensional_median_and_error_bar(x).median
                for x in self.priorkeys
            ]

        # make dummy frequency array
        self.dummy_freq = 10 ** (
            np.linspace(np.log10(self.fit_min / 1e6), np.log10(self.fit_max / 1e6), 100)
        )
        self.best_fit_function = self.__SED_func__(self.dummy_freq, *self.fit_params)
        return self.best_fit_function

    def get_fit_range_funcs(self, num=25):
        if not hasattr(self, "dummy_freq"):
            self.dummy_freq = 10 ** (
                np.linspace(
                    np.log10(self.fit_min / 1e6), np.log10(self.fit_max / 1e6), 100
                )
            )
        # get 25 realisations of the posterior
        self.randidx = np.random.randint(
            0, self.result.posterior[self.priorkeys[0]].values.shape[0], size=(num, 1)
        )
        self.posterior_list = [
            np.array([self.result.posterior[x].values[self.randidx]])
            for x in self.priorkeys
        ]
        self.best_fit_range = self.__SED_func__(
            self.dummy_freq.reshape(1, -1), *self.posterior_list
        )
        return self.best_fit_range[0]

    def get_posterior_param_draws(self, num=25):
        self.randidx = np.random.randint(
            0, self.result.posterior[self.priorkeys[0]].values.shape[0], size=(num, 1)
        )
        self.posterior_list = [
            np.array([self.result.posterior[x].values[self.randidx]])
            for x in self.priorkeys
        ]
        return self.posterior_list

    def get_param_medians_errors(self):
        """returns the median and 68% CI (equivalent approx. to 1 sigma error) from the bilby result"""
        if not hasattr(self, "result"):
            raise AttributeError(
                "Cannot get parameter intervals before running the sampler! Try calling run_sampler() first."
            )

        self.fit_param_ranges = [
            self.result.get_one_dimensional_median_and_error_bar(x)
            for x in self.priorkeys
        ]
        return self.fit_param_ranges

    def get_params(self):
        """Returns a list whose keys are the parameters of the model, and values are empty"""
        return self.priorkeys
