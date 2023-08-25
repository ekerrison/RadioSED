#!/usr/bin/python
import sys
import os
import numpy as np
from types import FunctionType
from inspect import signature

#bilby for mcmc fitting!
import bilby
import george
from scipy.linalg import cholesky, cho_solve
from RaiSEDModel import RaiSEDModel
from helper_functions import get_credible_interval, get_retrig_intervals
####################################################################################################################
#                                        Emily Kerrison (SIFA 2022)                                                #
#               Based heavily on James Allison's SED fitting code from 2014, this function fits                    #
#               a radio SED with the function described in Snellen+1997. The model is for a synchrotron            #
#               emission power law with low-freq absorption. It is fit in linear space. The code then              #
#               outputs the reduced chi-squared, as well as two information criteria, AIC () and BIC ()            #
#               which can be used to distinguish between this and other models to determine which                  #
#               best characterises the SED. It assumes input frequencies in Hz.                                    #
####################################################################################################################

class RaiSEDGPModel(RaiSEDModel):
    '''
    Generic Gaussian Process modelling class for my SED code
    '''

    #initial function
    def __init__(self, george_model, george_model_defaults, **kwargs):
        super().__init__(**kwargs)
        self.gp = True
        self.george_model = george_model
        self.george_model_defaults = george_model_defaults
        self.gp_params = 2
        self.md = len(signature(self.__SED_func__).parameters)-1 + self.gp_params # additional 2 for gp kernel


    def initialise_kernel(self):
        self.gp_kernel =  0.2*george.kernels.Matern32Kernel(5.0, block = (73,230)) # add some other kernel to ensure smoothness around the block!+ george.kernels.ConstantKernel(log_constant = 0)
        return

    #setting the likelihood to a bilby likelihood object using the appropriate SED function
    def set_likelihood(self):
        self.likelihood = bilby.core.likelihood.GeorgeLikelihood(kernel = self.gp_kernel, mean_model = self.george_model(*self.george_model_defaults), t = self.freq, y = self.flux, yerr = self.ferr)
        return

    def setup_sampler(self, prior: bilby.core.prior.dict.PriorDict, **kwargs):
        ''' Wrapper function for calling set_prior, initialise_kernel, set_likelihood and set_sampler'''
        self.set_prior(prior)
        self.initialise_kernel()
        self.set_likelihood()
        self.set_sampler(**kwargs)
        return

    def get_bic(self):
        if not hasattr(self, fit_params_func):
            self.fit_params_func = [self.result.get_one_dimensional_median_and_error_bar(x).median for x in self.priorkeys[:-2]]
        if not hasattr(self, fit_params_noise):
            self.fit_params_noise = [self.result.get_one_dimensional_median_and_error_bar(x).median for x in self.priorkeys[-2:]]

        self.final_kernel = np.exp(self.fit_params_noise[0])*george.kernels.Matern32Kernel(np.exp(self.fit_params_noise[1]), block = (73,230))

        #creating the GP model
        self.final_gp = george.GP(kernel = self.final_kernel, mean=self.__SED_func__(*self.fit_params_func))#, white_noise=white_noise_mod) #Joe used 0.2*matern kernel (not sure why?! CHECK!)

        #setting up the model using x values in MHz to get the covariance matrix
        self.final_gp.compute(self.freq, yerr = self.ferr)

        #get the covariance matrix cholesky decomposed for inversion
        self.cov_mat = self.final_gp.get_matrix(self.freq)

        #to add only the variance from the NOT gleam points - this will still add TXS uncertainty which I think we want
        #nogleam_ferr = ferr
        #nogleam_ferr[data['Survey quickname'] == 'GLEAM'] = 0
        #cov_mat += np.diag(nogleam_ferr)**2
        self.cov_mat += np.diag(self.ferr)**2

        #use cholesky decomposition for solving (speedier!)
        #cholesky decompose the covariance matrix for faster solving
        self.cov_mat = cholesky(self.cov_mat, overwrite_a=True, lower=False)

        #define x_modelled - x_obs in matrix notation
        self.residuals = lambda params: SED_func(self.freq, *params) - self.flux

        # Define chi-squared calculation - THIS MUST USE MATRIX NOTATION SINCE WE HAVE COVARIANCE!
        self.fit_chisq = np.dot(residuals(self.fit_params_func).T, cho_solve((self.cov_mat, False), residuals(self.fit_params_func)))

        #calculate information criteria
        fit_aic = fit_chisq + 2*len(fit_params)
        self.fit_bic = self.fit_chisq + len(self.fit_params_func)*np.log(len(self.freq))
        return self.fit_bic

    def get_best_fit_func(self):
        if not hasattr(self, 'fit_params_func'):
            self.fit_params_func = [self.result.get_one_dimensional_median_and_error_bar(x).median for x in self.priorkeys[:-self.gp_params]]

        #make dummy frequency array
        self.dummy_freq = 10 ** (np.linspace(np.log10(self.fit_min/1e6), np.log10(self.fit_max/1e6), 100) )
        self.best_fit_function = self.__SED_func__(self.dummy_freq, *self.fit_params_func)


        return self.best_fit_function

    def get_fit_range_funcs(self):
        '''returns the median and 68% CI (equivalent approx. to 1 sigma error) from the bilby result'''
        if not hasattr(self, 'result'):
            raise AttributeError('Cannot get parameter intervals before running the sampler! Try calling run_sampler() first.')

        if not hasattr(self, 'dummy_freq'):
            self.dummy_freq = 10 ** (np.linspace(np.log10(self.fit_min/1e6), np.log10(self.fit_max/1e6), 100) )
        #get 25 realisations of the posterior
        self.randidx = np.random.randint(0, self.result.posterior[self.priorkeys[0]].values.shape[0], size = (25,1))
        self.posterior_list = [np.array([self.result.posterior[x].values[self.randidx]]) for x in self.priorkeys[:-self.gp_params]]
        self.fit_param_ranges =  self.__SED_func__(self.dummy_freq.reshape(1,-1), *self.posterior_list)
        return self.fit_param_ranges[0]
