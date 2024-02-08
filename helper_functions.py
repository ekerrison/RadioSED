#!/usr/bin/python
import sys
import os
import warnings
import numpy as np
import pandas as pd


# bilby for likelihoods
from bilby import likelihood
import george
import inspect

# for gp fitting
from scipy.linalg import cholesky, cho_solve
from scipy.special import erf


####################################################################
# SED functions
# Define function to model power law SED
def linear_sed_func(x, C, alpha):
    y = np.zeros(len(np.atleast_1d(x)))

    y = C * (x**alpha)

    return y


def orienti_sed_func(x, a, b, c):
    """
    Parabola in log-space
    a: curvature
    b: linear param
    c: quadratic param
    """
    y = np.zeros(len(np.atleast_1d(x)))
    y = 10 ** (a + np.log10(x) * (b + c * np.log10(x)))
    return y


def snellen_sed_func(x, a, b, c, d):
    """
    Generic peaked model with power laws either side of the peak
    a: peak frequency
    b: peak flux
    c: alpha_thick
    d: alpha_thin
    """
    y = (
        b
        / (1.0 - np.exp(-1.0))
        * ((x / a) ** c)
        * (1.0 - np.exp(-((x / a) ** (d - c))))
    )
    return y


def retriggered_sed_func(x, a, b, c, d, S_norm, alpha):
    """
    Parameters:
        a = peak frequency (MHz)
        b = peak flux (Jy)
        c = alpha_thick (spectral index in thick region below the peak)
        d = alpha_thin (spectral index in thin region above the peak)
        S_norm = normalisation factor for the additional power law bit
        alpha = spectral index of the additional power law bit
    """
    y = np.zeros(len(np.atleast_1d(x)))

    y = (
        b
        / (1.0 - np.exp(-1.0))
        * ((x / a) ** c)
        * (1.0 - np.exp(-((x / a) ** (d - c))))
        + S_norm * x**alpha
    )

    return y


####################################################################
# Classes of the above for Gaussian process fitting
class LinearModel(george.modeling.Model):
    parameter_names = ("C", "alpha")

    def get_value(self, t):
        return self.C * (t.flatten() ** self.alpha)


class OrientiModel(george.modeling.Model):
    parameter_names = ("a", "b", "c")

    def get_value(self, t):
        return 10 ** (
            self.a + np.log10(t.flatten()) * (self.b + self.c * np.log10(t.flatten()))
        )


class SnellenModel(george.modeling.Model):
    parameter_names = ("a", "b", "c", "d")

    def get_value(self, t):
        return (
            self.b
            / (1.0 - np.exp(-1.0))
            * ((t.flatten() / self.a) ** self.c)
            * (1.0 - np.exp(-((t.flatten() / self.a) ** (self.d - self.c))))
        )


class RetriggeredModel(george.modeling.Model):
    parameter_names = ("a", "b", "c", "d", "S_norm", "alpha")

    def get_value(self, t):
        return (
            self.b
            / (1.0 - np.exp(-1.0))
            * ((t.flatten() / self.a) ** self.c)
            * (1.0 - np.exp(-((t.flatten() / self.a) ** (self.d - self.c))))
            + self.S_norm * t.flatten() ** self.alpha
        )


####################################################################
# other useful functions


def get_param_strs_toplot(model):
    """Takes a RaiSEDModel objects and returns the strings outlining the parameter
    values to be added to a plot

    Parameters
    ==========
    model: an input RaiSEDModel object that has been run through bilby

    Returns
    ==========
    plot_str_list: a list of strings to be put on the plot, 1 string per parameter
    """
    plot_str_list = []

    if model.gp:
        prefix = "mean:"
    else:
        prefix = ""

    # linear
    if "PL" in model.model_type:
        fit_param_intervals = model.get_param_medians_errors()

        # prior order is C, alpha
        c_str = " S$_{{norm}}= {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
            fit_param_intervals[0].median,
            fit_param_intervals[0].plus,
            fit_param_intervals[0].minus,
        )
        alpha_str = "$\\alpha = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
            fit_param_intervals[1].median,
            fit_param_intervals[1].plus,
            fit_param_intervals[1].minus,
        )
        plot_str_list = [c_str, alpha_str]

    # orienti - put this back in!!
    elif "curved" in model.model_type:
        if not model.param_intervals:
            raise AttributeError(
                '{} is the best model but its intervals have not been calculated yet, please call "get_orienti_intervals(model_object)" before plotting'.format(
                    model.model_type
                )
            )
        # prior order is a, b, c
        peak_freq_str = "$\\nu_{{peak}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ MHz".format(
            model.param_intervals["peak_freq_interval"][0],
            model.param_intervals["peak_freq_interval"][2],
            model.param_intervals["peak_freq_interval"][1],
        )
        peak_flux_str = "$S_{{peak}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ Jy".format(
            model.param_intervals["peak_flux_interval"][0],
            model.param_intervals["peak_flux_interval"][2],
            model.param_intervals["peak_flux_interval"][1],
        )
        alpha_thick_str = "$\\alpha_{{thick}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
            model.param_intervals["alpha_thick_interval"][0],
            model.param_intervals["alpha_thick_interval"][2],
            model.param_intervals["alpha_thick_interval"][1],
        )
        alpha_thin_str = "$\\alpha_{{thin}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
            model.param_intervals["alpha_thin_interval"][0],
            model.param_intervals["alpha_thin_interval"][2],
            model.param_intervals["alpha_thin_interval"][1],
        )
        plot_str_list = [peak_freq_str, peak_flux_str, alpha_thick_str, alpha_thin_str]

    # snellen
    elif "PS" in model.model_type:
        fit_param_intervals = model.get_param_medians_errors()

        # prior order is a,b,c,d
        peak_freq_str = "$\\nu_{{peak}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ MHz".format(
            fit_param_intervals[0].median,
            fit_param_intervals[0].plus,
            fit_param_intervals[0].minus,
        )
        peak_flux_str = "$S_{{peak}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ Jy".format(
            fit_param_intervals[1].median,
            fit_param_intervals[1].plus,
            fit_param_intervals[1].minus,
        )
        alpha_thick_str = "$\\alpha_{{thick}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
            fit_param_intervals[2].median,
            fit_param_intervals[2].plus,
            fit_param_intervals[2].minus,
        )
        alpha_thin_str = "$\\alpha_{{thin}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
            fit_param_intervals[3].median,
            fit_param_intervals[3].plus,
            fit_param_intervals[3].minus,
        )
        plot_str_list = [peak_freq_str, peak_flux_str, alpha_thick_str, alpha_thin_str]

    # retriggered
    elif "retrig" in model.model_type:
        # use our other helper function which should have already been called!!
        if not model.param_intervals:
            raise AttributeError(
                '{} is the best model but its intervals have not been calculated yet, please call "get_retrig_intervals(model_object)" before plotting'.format(
                    model.model_type
                )
            )
        # peak_freq_interval, peak_flux_interval, trough_freq_interval, trough_flux_interval, alpha_retrig_interval, \
        # alpha_thick_interval, alpha_thin_interval, func_type = get_retrig_intervals(result = model, SED_func = retriggered_sed_func, gp = model.gp, min_obs_freq = min_obs_freq, max_obs_freq = max_obs_freq)

        # prior order is a,b,c,d,Snorm,alpha
        peak_freq_str = "$\\nu_{{peak}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ MHz".format(
            model.param_intervals["peak_freq_interval"][0],
            model.param_intervals["peak_freq_interval"][2],
            model.param_intervals["peak_freq_interval"][1],
        )
        peak_flux_str = "$S_{{peak}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ Jy".format(
            model.param_intervals["peak_flux_interval"][0],
            model.param_intervals["peak_flux_interval"][2],
            model.param_intervals["peak_flux_interval"][1],
        )
        trough_freq_str = "$\\nu_{{min}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ MHz".format(
            model.param_intervals["trough_freq_interval"][0],
            model.param_intervals["trough_freq_interval"][2],
            model.param_intervals["trough_freq_interval"][1],
        )
        trough_flux_str = "$S_{{min}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ Jy".format(
            model.param_intervals["trough_flux_interval"][0],
            model.param_intervals["trough_flux_interval"][2],
            model.param_intervals["trough_flux_interval"][1],
        )
        alpha_retrig_str = (
            "$\\alpha_{{retrig}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
                model.param_intervals["alpha_retrig_interval"][0],
                model.param_intervals["alpha_retrig_interval"][2],
                model.param_intervals["alpha_retrig_interval"][1],
            )
        )
        alpha_thick_str = "$\\alpha_{{thick}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
            model.param_intervals["alpha_thick_interval"][0],
            model.param_intervals["alpha_thick_interval"][2],
            model.param_intervals["alpha_thick_interval"][1],
        )
        alpha_thin_str = "$\\alpha_{{thin}} = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
            model.param_intervals["alpha_thin_interval"][0],
            model.param_intervals["alpha_thin_interval"][2],
            model.param_intervals["alpha_thin_interval"][1],
        )
        plot_str_list = [
            peak_freq_str,
            peak_flux_str,
            trough_freq_str,
            trough_flux_str,
            alpha_retrig_str,
            alpha_thick_str,
            alpha_thin_str,
        ]

    else:
        raise TypeError(
            "I cannot find this type of model to extract parameter strings. (Yikes!)"
        )

    return plot_str_list


def get_best_model(array):
    """Takes an array of RaiSEDModel objects and orders them so that the best-fitting
    model is first, i.e. in decreasing logZ order.

    NOTE: This code adopts the Jeffrey's scale as slightly modified by Kass & Rafferty (1995)
    to aid in model selection:

    0 < log10 B12 < 0.5 --> marginal preference for model 1 (<~ 3 times more likely)
    0.5 < log10 B12 < 1 --> preference for model 1 (3 < x < 10 times more likely)
    1 < log10 B12 < 2 --> strong preference for model 1 (10 < x < 100 times more likely)
    log10 B12 > 2 --> very strong preference for model 1 (> 100 times more likely)

    Parameters
    ==========
    array: an input array of RaiSEDModel objects that have been run through bilby

    Returns
    ==========
    sorted_array: the input array sorted so that the best fitting model comes first
    """

    log10Z_arr = np.array([x.get_log10z()[0] for x in array])
    log10Z_err_arr = np.array([x.get_log10z()[1] for x in array])
    log10Z_noise_arr = np.array([x.get_log10z()[2] for x in array])
    bmd_array = np.array([x.get_bmd() for x in array])
    md_array = np.array([x.get_md() for x in array])

    # now sort them
    bmd_array = bmd_array[np.argsort(log10Z_arr)[::-1]]
    md_array = md_array[np.argsort(log10Z_arr)[::-1]]
    array = array[np.argsort(log10Z_arr)[::-1]]
    log10Z_err_arr = log10Z_err_arr[np.argsort(log10Z_arr)[::-1]]
    log10Z_noise_arr = log10Z_noise_arr[np.argsort(log10Z_arr)[::-1]]
    log10Z_arr = log10Z_arr[np.argsort(log10Z_arr)[::-1]]

    # print('EVIDENCE, NOISE AND ERROR ARRAYS:')
    # print(log10Z_arr)
    # print(log10Z_noise_arr)
    # print(log10Z_err_arr)

    # if the log10 of the evidence error is > 1 then the fit is suspicious and (from visual inspection)
    # likely bimodal

    # now look at the first model, and see what the bayes factor is!
    bf1 = log10Z_arr[0] - log10Z_arr[1]
    bf1_lim = log10Z_arr[0]
    # print('bayes factor: {} and dimensional difference: {}'.format(bf1,  (md_array[0] - md_array[1])))
    # if there is a marginal prefernce for model 1 and model 2 has fewer dimensions
    # according to the Bayesian Model Dimensionality, call model 2 the best fit
    if (
        bf1 > 0 and (md_array[0] - md_array[1]) > 0
    ):  # make sure that the second model is simpler
        if (
            bf1 < 0.5
        ):  # and bmd_array[1] < bmd_array[0]: #np.log10(md_array[0] - md_array[1] + 1):
            # swap models 1 and 2, since the evidence for 1 over 2 is marginal
            # and 2 is the simpler model
            log10Z_arr[[0, 1]] = log10Z_arr[[1, 0]]
            bmd_array[[0, 1]] = bmd_array[[1, 0]]
            md_array[[0, 1]] = md_array[[1, 0]]
            array[[0, 1]] = array[[1, 0]]

    # now do the same but with the first and third models, in case that is preferred!
    if log10Z_arr.shape[0] > 2:
        bf2 = log10Z_arr[0] - log10Z_arr[2]
        if (
            bf2 > 0 and (md_array[0] - md_array[2]) > 0
        ):  # make sure that the third model is simpler
            if (
                bf2 < 0.5
            ):  # and bmd_array[2] < bmd_array[0]: #np.log10(md_array[0] - md_array[2] + 1):
                log10Z_arr[[0, 2]] = log10Z_arr[[2, 0]]
                bmd_array[[0, 2]] = bmd_array[[2, 0]]
                md_array[[0, 2]] = md_array[[2, 0]]
                array[[0, 2]] = array[[2, 0]]

    return array, log10Z_arr, bmd_array


def get_credible_interval(array, interval=np.nan, lower_quant=0.16, upper_quant=0.84):
    """Calculate the median and error bar for a given array

    Parameters
    ==========
    array:
        The data for which to calculate the median and error bar
    interval: float [0,1]
        An interval for which to calculate the error bars, can be used as
        an alternative to specifying lower_quant and upper_quant
    lower_quant: float
        The lower quantile to calculate the error bars for.
    upper_quant: float
        The lower quantile to calculate the error bars for.

    The default is a 68% credible interval (which can be thought of as
    1 sigma), corresponding to lower_quant = 0.16, upper_quant = 0.84

    Returns
    =======
    summary: list
        A list of: median, lower, upper
    """

    # if interval specified, calculate lower and upper quantiles
    if not np.isnan(interval):
        if interval >= 1 or interval <= 0:
            raise ValueError("interval must be between 0 and 1")

        # convert to quantiles
        lower_quant = (1 - interval) / 2
        upper_quant = 1 - lower_quant

    # remove nans from array
    array = array[~np.isnan(array)]
    # now get the quantiles!
    quants_to_compute = np.array([lower_quant, 0.5, upper_quant])
    quants = np.percentile(array, quants_to_compute * 100)
    median = quants[1]
    plus = quants[2] - median
    minus = median - quants[0]

    return [median, minus, plus]


def get_orienti_intervals(
    result,
    SED_func,
    gp=False,
    freq_min=1e7,
    freq_max=1e11,
    bilby=True,
    min_obs_freq=np.nan,
    max_obs_freq=np.nan,
):
    """
    gets the confidence intervals for:
    the peak frequency and corresponding flux
    a linear approximation of the spectral index either side of the turning point
    ordered as:
        alpha1 = alpha_below_peak
        alpha2 = alpha_above_peak

    Returns
    ==========
    A list comprising the following elements:
        peak_freq_interval = [val, p, m]
        peak_flux_interval = [val, p, m]
        alpha_thick_interval = [val, p, m]
        alpha_thin_interval = [val, p, m]
        func_type = str

    """

    if gp:
        prefix = "mean:"
    else:
        prefix = ""

    print(min_obs_freq, max_obs_freq)

    # a hack so this function works with both bilby and jaxns
    if bilby:
        # parameter posteriors
        a_result = result.result.posterior[prefix + "a"].values
        b_result = result.result.posterior[prefix + "b"].values
        c_result = result.result.posterior[prefix + "c"].values
    else:
        a_result = result[prefix + "a"]
        b_result = result[prefix + "b"]
        c_result = result[prefix + "c"]

    # get physical params, where peak_freq = -b/2*c and peak_flux = y_val at peak_Freq
    peak_freq_dist = 10 ** (-b_result / (2 * c_result))
    peak_flux_dist = SED_func(peak_freq_dist, a_result, b_result, c_result)

    # get median and stdev, as well as upper and lower errors (68% credible intervals)
    freq_credible_interval = get_credible_interval(peak_freq_dist)
    [peak_freq_med, peak_freq_plus, peak_freq_minus] = freq_credible_interval

    flux_credible_interval = get_credible_interval(peak_flux_dist)
    [peak_flux_med, peak_flux_plus, peak_flux_minus] = flux_credible_interval

    # get set of points to do spectral indices on: We are fitting in MHz!!
    freq_arr = 10 ** (
        np.linspace(np.log10(freq_min / 1e6), np.log10(freq_max / 1e6), 1000)
    )  # np.linspace(np.log10(70), np.log10(30e3)
    min_obs_idx = np.argmin(np.abs(freq_arr - min_obs_freq / 1e6))
    max_obs_idx = np.argmin(np.abs(freq_arr - max_obs_freq / 1e6))

    # broadcast it for later use
    freq_arr_broadcast = np.tile(freq_arr, (a_result.shape[0], 1))

    # get the fluxes for each combination of parameters in the posterior
    flux_dist = SED_func(freq_arr.reshape(-1, 1), a_result, b_result, c_result)
    flux_dist = flux_dist.transpose()

    # now get the alpha_thick and alpha_thin indices away from the peak!
    # taking index at last observed point, following an old snellen paper (FIND REFERENCE!) where they did this
    alpha_thick_grads = 2 * c_result * np.log10(freq_arr_broadcast[:, min_obs_idx]) + b_result
    alpha_thick_interval = get_credible_interval(alpha_thick_grads)

    # thin_idxs = np.around(freq_arr.shape[0] + (freq_arr.shape[0] - peak_idxs)*0.5)
    # alpha_thin_grads = 2*c_result*flux_dist[thin_idxs] + b_result
    alpha_thin_grads = 2 * c_result * np.log10(freq_arr_broadcast[:, max_obs_idx]) + b_result
    alpha_thin_interval = get_credible_interval(alpha_thin_grads)

    # write dict to result object
    result.param_intervals["peak_freq_interval"] = freq_credible_interval
    result.param_intervals["peak_flux_interval"] = flux_credible_interval
    result.param_intervals["alpha_thick_interval"] = alpha_thick_interval
    result.param_intervals["alpha_thin_interval"] = alpha_thin_interval
    return (
        freq_credible_interval,
        flux_credible_interval,
        alpha_thick_interval,
        alpha_thin_interval,
    )


def get_retrig_intervals(
    result,
    SED_func,
    gp=False,
    freq_min=1e7,
    freq_max=1e11,
    bilby=True,
    min_obs_freq=np.nan,
    max_obs_freq=np.nan,
):
    """
    gets the confidence intervals for:
    the peak frequency and corresponding flux
    the trough (local minimum) and corresponding flux
    a linear approximation of the spectral index either side of the turning points, and in between them,
    ordered as:
        alpha1 = alpha_retriggered
        alpha2 = alpha_below_peak
        alpha3 = alpha_above_peak

    Returns
    ==========
    A list comprising the following elements:
        peak_freq_interval = [val, p, m]
        peak_flux_interval = [val, p, m]
        trough_freq_interval = [val, p, m]
        trough_flux_interval = [val, p, m]
        alpha_retrig_interval = [val, p, m]
        alpha_thick_interval = [val, p, m]
        alpha_thin_interval = [val, p, m]
        func_type = str

    """
    if gp:
        prefix = "mean:"
    else:
        prefix = ""


    # a hack so this function works with both bilby and jaxns
    if bilby:
        # parameter posteriors
        a_result = result.result.posterior[prefix + "a"].values
        b_result = result.result.posterior[prefix + "b"].values
        c_result = result.result.posterior[prefix + "c"].values
        d_result = result.result.posterior[prefix + "d"].values
        s_norm_result = result.result.posterior[prefix + "S_norm"].values
        alpha_result = result.result.posterior[prefix + "alpha"].values
    else:
        a_result = result[prefix + "a"]
        b_result = result[prefix + "b"]
        c_result = result[prefix + "c"]
        d_result = result[prefix + "d"]
        s_norm_result = result[prefix + "snorm"]
        alpha_result = result[prefix + "alpha"]

    # get physical param distributions! We are fitting in MHz!!
    freq_arr = 10 ** (
        np.linspace(np.log10(freq_min / 1e6), np.log10(freq_max / 1e6), 1000)
    )  # np.linspace(np.log10(70), np.log10(30e3)

    # broadcast it for later use
    freq_arr_broadcast = np.tile(freq_arr, (a_result.shape[0], 1))

    # get the fluxes for each combination oxf parameters in the posterior
    flux_dist = SED_func(
        freq_arr.reshape(-1, 1),
        a_result,
        b_result,
        c_result,
        d_result,
        s_norm_result,
        alpha_result,
    )
    flux_dist = flux_dist.transpose()

    #mask out things below 10-5 because we run into jumps due to the machine
    #floor down here
    if flux_dist[flux_dist < 10e-5].shape[0] > 0:
        unique_keys, indices = np.unique(np.argwhere(flux_dist < 10e-5)[:,0], return_index=True)

        freq_max_new = freq_arr[np.min(np.argwhere(flux_dist < 20e-5)[indices, 1])]
        freq_max_new = 1e6*freq_max_new

        #print('new max freq: {}GHz'.format(freq_max_new/1e6))
        if freq_max_new/1e6 < 500:
            freq_max_new = freq_arr[np.max(np.argwhere(flux_dist < 20e-5)[indices, 1])]
            freq_max_new = 1e6*freq_max_new
        #print('new max freq (second try): {}GHz'.format(freq_max_new/1e6))
        freq_arr = 10 ** (
            np.linspace(np.log10(freq_min / 1e6), np.log10(freq_max_new / 1e6), 1000)
        )  # np.linspace(np.log10(70), np.log10(30e3)

        # broadcast it for later use
        freq_arr_broadcast = np.tile(freq_arr, (a_result.shape[0], 1))

        # get the fluxes for each combination of parameters in the posterior
        flux_dist = SED_func(
            freq_arr.reshape(-1, 1),
            a_result,
            b_result,
            c_result,
            d_result,
            s_norm_result,
            alpha_result,
        )
        flux_dist = flux_dist.transpose()

    #extra check to make sure this is really gone
    '''
    print('number below error threshold:')
    print(flux_dist[flux_dist < 10e-5].shape[0])
    
    freq_max_loop = freq_max
    while flux_dist[flux_dist < 10e-5].shape[0] > 0:
        print(flux_dist[flux_dist < 10e-5].shape[0])
        # get physical param distributions! We are fitting in MHz!!
        freq_max_loop *= 0.9
        print(freq_max_loop)
        freq_arr = 10 ** (
            np.linspace(np.log10(freq_min / 1e6), np.log10(freq_max_loop / 1e6), 1000)
        )  # np.linspace(np.log10(70), np.log10(30e3)

        # broadcast it for later use
        freq_arr_broadcast = np.tile(freq_arr, (a_result.shape[0], 1))

        # get the fluxes for each combination of parameters in the posterior
        flux_dist = SED_func(
            freq_arr.reshape(-1, 1),
            a_result,
            b_result,
            c_result,
            d_result,
            s_norm_result,
            alpha_result,
        )
        flux_dist = flux_dist.transpose()
    '''

    # use np.diff to find turning pts
    diff_dist = np.diff(flux_dist, axis=1)

    # get the sign of each point
    diff_sign = np.sign(diff_dist)

    # THIS IS TO IGNORE INFLEXION POINTS!! Added 09/09
    diff_sign[diff_sign == 0] = 1

    # look for a sign change
    signchange = ((diff_sign - np.roll(diff_sign, 1, axis=1)) > 0).astype(int)
    signchange -= ((diff_sign - np.roll(diff_sign, 1, axis=1)) < 0).astype(int)

    # make the first row zero
    signchange[0, :] = 0

    # add on a column to keep it the same size!
    signchange = np.c_[signchange, np.zeros(a_result.shape)]

    # get turning pts, where signchange = 1 in each row
    local_min_idxs = np.where(signchange > 0)
    local_max_idxs = np.where(signchange < 0)

    '''
    print(flux_dist.shape)
    print(freq_arr_broadcast.shape)
    print(local_min_idxs[0].shape)
    print(local_max_idxs[0].shape)
    '''

    # get max and min fluxes based on this
    max_fluxes = flux_dist[local_max_idxs]
    min_fluxes = flux_dist[local_min_idxs]

    # get max and min corresponding frequencies
    max_freqs = freq_arr_broadcast[local_max_idxs]
    min_freqs = freq_arr_broadcast[local_min_idxs]

    # now get median values and errorbars (68% credible intervals)
    try:
        peak_freq_interval = get_credible_interval(max_freqs)
        peak_flux_interval = get_credible_interval(max_fluxes)
    except IndexError as e:
        peak_freq_interval = [-1, 0, 0]
        peak_flux_interval = [-1, 0, 0]

    try:
        trough_freq_interval = get_credible_interval(min_freqs)
        trough_flux_interval = get_credible_interval(min_fluxes)
    except IndexError as e:
        trough_freq_interval = [-1, 0, 0]
        trough_flux_interval = [-1, 0, 0]

    # now get interval idxs, make it simple and do 30% away from the max/min on either side
    distance_factor = 0.3  # per cent
    max_lower_idx = np.argmin(
        np.abs(freq_arr - peak_freq_interval[0] * (1 - distance_factor))
    )
    max_upper_idx = np.argmin(
        np.abs(freq_arr - peak_freq_interval[0] * (1 + distance_factor))
    )

    min_lower_idx = np.argmin(
        np.abs(freq_arr - trough_freq_interval[0] * (1 - distance_factor))
    )
    min_upper_idx = np.argmin(
        np.abs(freq_arr - trough_freq_interval[0] * (1 + distance_factor))
    )

    # find if the peak is above or below the trough in frequency
    # print('peak and trough intervals')
    # print(peak_freq_interval)
    # print(trough_freq_interval)

    # print('min and max obs')
    # print(min_obs_freq, max_obs_freq)
    # print(min_obs_freq, max_obs_freq)

    # print(np.min(freq_arr), np.max(freq_arr))

    # if peak_freq_interval[0] < min_obs_freq/1e6 or peak_freq_interval[0] > max_obs_freq/1e6:
    #    peak_freq_interval = [-1,0,0]
    #    peak_flux_interval = [-1,0,0]

    # if trough_freq_interval[0] < min_obs_freq/1e6:
    #    trough_freq_interval = [-1,0,0]
    #    trough_flux_interval = [-1,0,0]

    if min_lower_idx == 0:
        distance_factor = 0.1  # x100 = per cent
        min_lower_idx = np.argmin(
            np.abs(freq_arr - trough_freq_interval[0] * (1 - distance_factor))
        )

    if max_upper_idx == len(freq_arr) - 1:
        distance_factor = 0.1  # x100 = per cent
        max_upper_idx = np.argmin(
            np.abs(freq_arr - peak_freq_interval[0] * (1 + distance_factor))
        )

    # if it's still zero then make it even smaller!
    # if min_lower_idx == 0:
    #    min_lower_idx = 1

    # if max_upper_idx == len(freq_arr) - 1:
    #     max_upper_idx = len(freq_arr) - 2

    #if max_lower = min_upper shift them apart a bit
    if max_lower_idx == min_upper_idx:
        max_lower_idx -= 3
        min_upper_idx += 3

    if max_lower_idx == 0:
        max_lower_idx += 1

    '''
    print(
        "calculating indices at: {:.2f}, {:.2f}, {:.2f}, {:.2f} MHz".format(
            freq_arr[max_lower_idx],
            freq_arr[max_upper_idx],
            freq_arr[min_lower_idx],
            freq_arr[min_upper_idx],
        )
    )
    '''

    #if we have no turning pts, asusme linear and just return the spectral index
    if (min_lower_idx == max_upper_idx and min_lower_idx == 0) or (max_lower_idx == min_lower_idx and max_upper_idx == min_upper_idx):
        alphagrads = (
            np.log10(flux_dist[:, 0])
            - np.log10(flux_dist[:, -1])
        ) / (
            np.log10(freq_arr_broadcast[:, 0])
            - np.log10(freq_arr_broadcast[:, -1])
        )

        # get the credible interval based on this
        alpha_interval = get_credible_interval(alphagrads)
        alpha_retrig_interval = [-1, 0, 0]
        alpha_thick_interval = [-1, 0, 0]
        alpha_thin_interval = alpha_interval
        func_type = 'linear'
        # write dict to result object
        result.param_intervals["peak_freq_interval"] = [-1, 0, 0]
        result.param_intervals["peak_flux_interval"] = [-1, 0, 0]
        result.param_intervals["trough_freq_interval"] = [-1, 0, 0]
        result.param_intervals["trough_flux_interval"] = [-1, 0, 0]
        result.param_intervals["alpha_retrig_interval"] = [-1, 0, 0]
        result.param_intervals["alpha_thick_interval"] = [-1, 0, 0]
        result.param_intervals["alpha_thin_interval"] = alpha_interval

        return (
            peak_freq_interval,
            peak_flux_interval,
            trough_freq_interval,
            trough_flux_interval,
            alpha_retrig_interval,
            alpha_thick_interval,
            alpha_thin_interval,
            func_type,
        )

    # print(trough_freq_interval[0], trough_freq_interval[0]*(1-distance_factor), freq_arr)

    # peak above trough
    if (
        peak_freq_interval[0] > trough_freq_interval[0]
        and peak_freq_interval[0] > freq_min / 1e6
        and trough_freq_interval[0] > freq_min / 1e6
    ):
        func_type = "minfirst"

        # go from min to max
        # alphathickgrads = (np.log10(flux_dist[local_max_idxs]) - np.log10(flux_dist[local_min_idxs]))/(np.log10(freq_arr_broadcast[local_max_idxs]) - np.log10(freq_arr_broadcast[local_min_idxs]))
        alphathickgrads = (
            np.log10(flux_dist[:, max_lower_idx])
            - np.log10(flux_dist[:, min_upper_idx])
        ) / (
            np.log10(freq_arr_broadcast[:, max_lower_idx])
            - np.log10(freq_arr_broadcast[:, min_upper_idx])
        )

        # get the credible interval based on this
        alpha_thick_interval = get_credible_interval(alphathickgrads)

        # calculate the gradient in each realisation
        # retriggrads = (np.log10(flux_dist[local_min_idxs]) - np.log10(flux_dist[local_min_idxs[0],0]))/(np.log10(freq_arr_broadcast[local_min_idxs]) - np.log10(freq_arr_broadcast[local_min_idxs[0],0]))
        retriggrads = (
            np.log10(flux_dist[:, min_lower_idx]) - np.log10(flux_dist[:, 0])
        ) / (
            np.log10(freq_arr_broadcast[:, min_lower_idx])
            - np.log10(freq_arr_broadcast[:, 0])
        )

        print("lower idx: {}".format(min_lower_idx))

        # get the credible interval based on this
        alpha_retrig_interval = get_credible_interval(retriggrads)

        # calculate the gradient in each realisation
        # thingrads = (np.log10(flux_dist[local_max_idxs[0],-1]) - np.log10(flux_dist[local_max_idxs]))/(np.log10(freq_arr_broadcast[local_max_idxs[0],-1]) - np.log10(freq_arr_broadcast[local_max_idxs]))
        thingrads = (
            np.log10(flux_dist[:, -1]) - np.log10(flux_dist[:, max_upper_idx])
        ) / (
            np.log10(freq_arr_broadcast[:, -1])
            - np.log10(freq_arr_broadcast[:, max_upper_idx])
        )

        # get the credible interval based on this
        alpha_thin_interval = get_credible_interval(thingrads)

    elif (
        peak_freq_interval[0] < trough_freq_interval[0]
        and peak_freq_interval[0] > 0
        and trough_freq_interval[0] > 0
    ):
        func_type = "maxfirst"

        # go from 0 to max
        # alphathickgrads = (np.log10(flux_dist[local_max_idxs]) - np.log10(flux_dist[local_max_idxs[0],0]))/(np.log10(freq_arr_broadcast[local_max_idxs]) - np.log10(freq_arr_broadcast[local_max_idxs[0],0]))
        alphathickgrads = (
            np.log10(flux_dist[:, max_lower_idx]) - np.log10(flux_dist[:, 0])
        ) / (
            np.log10(freq_arr_broadcast[:, max_lower_idx])
            - np.log10(freq_arr_broadcast[:, 0])
        )

        # get the credible interval based on this
        alpha_thick_interval = get_credible_interval(alphathickgrads)

        # go from local min to end
        # retriggrads = (np.log10(flux_dist[local_min_idxs[0],-1]) - np.log10(flux_dist[local_min_idxs]))/(np.log10(freq_arr_broadcast[local_min_idxs[0],-1]) - np.log10(freq_arr_broadcast[local_min_idxs]))
        retriggrads = (
            np.log10(flux_dist[:, -1]) - np.log10(flux_dist[:, min_upper_idx])
        ) / (
            np.log10(freq_arr_broadcast[:, -1])
            - np.log10(freq_arr_broadcast[:, min_upper_idx])
        )

        # get the credible interval based on this
        alpha_retrig_interval = get_credible_interval(retriggrads)

        # go from max to min
        # thingrads = (np.log10(flux_dist[local_min_idxs]) - np.log10(flux_dist[local_max_idxs]))/(np.log10(freq_arr_broadcast[local_min_idxs]) - np.log10(freq_arr_broadcast[local_max_idxs]))
        thingrads = (
            np.log10(flux_dist[:, max_upper_idx])
            - np.log10(flux_dist[:, min_lower_idx])
        ) / (
            np.log10(freq_arr_broadcast[:, max_upper_idx])
            - np.log10(freq_arr_broadcast[:, min_lower_idx])
        )

        # get the credible interval based on this
        alpha_thin_interval = get_credible_interval(thingrads)

    # we have only a trough and no peak
    elif peak_freq_interval[0] < 0 and trough_freq_interval[0] > 0:
        func_type = "minonly"
        # go from min to peak
        # alphathickgrads = (np.log10(flux_dist[local_min_idxs[0],-1]) - np.log10(flux_dist[local_min_idxs]))/(np.log10(freq_arr_broadcast[local_min_idxs[0],-1]) - np.log10(freq_arr_broadcast[local_min_idxs]))
        alphathickgrads = (
            np.log10(flux_dist[:, -1]) - np.log10(flux_dist[:, min_upper_idx])
        ) / (
            np.log10(freq_arr_broadcast[:, -1])
            - np.log10(freq_arr_broadcast[:, min_upper_idx])
        )

        # get the credible interval based on this
        alpha_thick_interval = get_credible_interval(alphathickgrads)

        # go from start to min
        # retriggrads = (np.log10(flux_dist[local_min_idxs]) - np.log10(flux_dist[local_min_idxs[0],0]))/(np.log10(freq_arr_broadcast[local_min_idxs]) - np.log10(freq_arr_broadcast[local_min_idxs[0],0]))
        retriggrads = (
            np.log10(flux_dist[:, min_lower_idx]) - np.log10(flux_dist[:, 0])
        ) / (
            np.log10(freq_arr_broadcast[:, min_lower_idx])
            - np.log10(freq_arr_broadcast[:, 0])
        )

        # get the credible interval based on this
        alpha_retrig_interval = get_credible_interval(retriggrads)

        # get the credible interval based on this
        alpha_thin_interval = [np.nan, 0, 0]

    # we have only a peak and no trough - shouldn't happen we should pick SNELLEN
    elif trough_freq_interval[0] <= freq_min / 1e6 and peak_freq_interval[0] > 0:
        warnings.warn("Only a peak no trough, despite being fit by a retriggered model")
        func_type = "maxonly"
        # go from start to peak
        # alphathickgrads = (np.log10(flux_dist[local_max_idxs]) - np.log10(flux_dist[local_max_idxs[0],0]))/(np.log10(freq_arr_broadcast[local_max_idxs]) - np.log10(freq_arr_broadcast[local_max_idxs[0],0]))
        alphathickgrads = (
            np.log10(flux_dist[:, max_lower_idx]) - np.log10(flux_dist[:, 0])
        ) / (
            np.log10(freq_arr_broadcast[:, max_lower_idx])
            - np.log10(freq_arr_broadcast[:, 0])
        )

        # print(alphathickgrads)
        # exit()
        # get the credible interval based on this
        alpha_thick_interval = get_credible_interval(alphathickgrads)

        # go from peak to end
        # thingrads = (np.log10(flux_dist[local_max_idxs[0],-1]) - np.log10(flux_dist[local_max_idxs]))/(np.log10(freq_arr_broadcast[local_max_idxs[0],-1]) - np.log10(freq_arr_broadcast[local_max_idxs]))
        thingrads = (
            np.log10(flux_dist[:, -1]) - np.log10(flux_dist[:, max_upper_idx])
        ) / (
            np.log10(freq_arr_broadcast[:, -1])
            - np.log10(freq_arr_broadcast[:, max_upper_idx])
        )

        # get the credible interval based on this
        alpha_thin_interval = get_credible_interval(thingrads)

        # get the credible interval based on this
        alpha_retrig_interval = [np.nan, 0, 0]

    elif trough_freq_interval[0] < 0 and peak_freq_interval[0] < 0:
        # it's basically linear!
        func_type = "linear"
        retriggrads = (np.log10(flux_dist[:, -1]) - np.log10(flux_dist[:, 0])) / (
            np.log10(freq_arr_broadcast[:, -1]) - np.log10(freq_arr_broadcast[:, 0])
        )

        # get the credible interval based on this
        alpha_retrig_interval = get_credible_interval(retriggrads)

        # get the credible interval based on this
        alpha_thin_interval = [-1, 0, 0]

        alpha_thick_interval = [np.nan, 0, 0]
    else:
        print(peak_freq_interval)
        print(peak_flux_interval)
        print("Something has gone wrong in fitting, please go back and check the bilby logs")
        exit()

    # write dict to result object
    result.param_intervals["peak_freq_interval"] = peak_freq_interval
    result.param_intervals["peak_flux_interval"] = peak_flux_interval
    result.param_intervals["trough_freq_interval"] = trough_freq_interval
    result.param_intervals["trough_flux_interval"] = trough_flux_interval
    result.param_intervals["alpha_retrig_interval"] = alpha_retrig_interval
    result.param_intervals["alpha_thick_interval"] = alpha_thick_interval
    result.param_intervals["alpha_thin_interval"] = alpha_thin_interval

    return (
        peak_freq_interval,
        peak_flux_interval,
        trough_freq_interval,
        trough_flux_interval,
        alpha_retrig_interval,
        alpha_thick_interval,
        alpha_thin_interval,
        func_type,
    )


def get_neighbour_density(survey_list, ra, dec):
    # catalogues from vizier
    # head_filepath = '/Users/eker0753/Documents/2022/
    head_filepath = "/Users/eker0753/Documents/2022/"
    filepath2 = head_filepath + "data/"
    fname = "included_surveys.csv"

    cat_data = pd.read_csv(
        filepath2 + fname, sep=",", header=0, skiprows=list(range(26, 58))
    )

    cat_data = cat_data[cat_data["bibcode"].isin(survey_list)]
    source_beams = cat_data["max beam size (arcsec)"].apply(
        lambda x: float(x.split(";")[0])
    )
    # sort values in cat
    source_beams = source_beams.sort_values(ascending=False)
    source_beams = source_beams.reset_index(drop=True)
    # get first (largest) value
    max_beam = source_beams.loc[0]

    # find how many other racs sources are within this beam size
    filepath3 = head_filepath + "data/radio_catalogues/racs/"
    isl_fname = "racs_island_full.csv"
    isl_data = pd.read_csv(filepath3 + isl_fname, header=0)

    # find how many islands are within the beam
    near_isls = isl_data[
        (isl_data["ra"] - ra) ** 2 + (isl_data["dec"] - dec) ** 2
        <= max_beam / (60 * 60)
    ]
    near_isls["dist"] = (near_isls["ra"] - ra) ** 2 + (near_isls["dec"] - dec) ** 2

    # so presumably one of these is the source itself
    neighbour_density = near_isls.shape[0] + 1

    return neighbour_density


def check_retrig_is_peaked(
    result,
    SED_func,
    gp=False,
    freq_min=1e7,
    freq_max=1e11,
    bilby=True,
    peak_freq=np.nan,
    trough_freq=np.nan,
):
    # externally in the fit_SED function we check whether the peak and trough our within the observed frequency range

    # check if peak and trough are within 250MHz, if so mark as not peaked!

    # now check if ???

    # now we need to check the spectral indices and decide if it looks peaked

    return


class GaussianCovLikelihood(likelihood.Analytical1DLikelihood):
    def __init__(self, x, y, function, sigma=None, a=None, tau=None):
        """
        A general Gaussian likelihood for known or unknown noise - the model
        parameters are inferred from the arguments of function

        Parameters
        ----------
        x, y: array-like
            The data to analyse
        function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        sigma: 2D array
            The covariance matrix of the data. A square array with length equal
            to the length of x and y. Must be given in Cholesky-factorised form!
        a  : the vertical scale o the matern kernel (a * kernels.Matern32Kernel(tau))
        tau: the length scale of the matern kernel (a * kernels.Matern32Kernel(tau))
        """
        self.x = x
        self.y = y
        self.N = len(x)
        self.sigma = sigma
        self.function = function

        # These lines of code infer the parameters from the provided function
        parameters = inspect.getargspec(function).args
        parameters.pop(0)
        super().__init__(parameters=dict.fromkeys(parameters))
        self.parameters = dict.fromkeys(parameters)

        self.function_keys = self.parameters.keys()
        if self.sigma is None:
            self.parameters["sigma"] = None

    def log_likelihood(self):
        sigma = self.parameters.get("sigma", self.sigma)
        model_parameters = {k: self.parameters[k] for k in self.function_keys}
        residuals = self.y - self.function(
            self.x, **model_parameters
        )  # ydata - SED_func(xdata)
        return -0.5 * np.dot(
            residuals.T, cho_solve((sigma, False), residuals)
        )  # -0.5*tranpose(R)*Inverse(covariance matrix)*R
        # plus a constant we don't care about!

#class for censored likelihood
class GaussianCensoredLikelihood(likelihood.GaussianLikelihood):
    def __init__(self, x, y, func, sigma=None, yUL=None, **kwargs):
        """
        A general Gaussian likelihood for known or unknown noise but with 
        censored data - the model parameters are inferred from the arguments of function

        Parameters
        ==========
        x, y: array_like
            The data to analyse
        yUL: array_like (boolean)
            Flags for whether the data is left censored (upper limits).
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        sigma: None, float, array_like
            If None, the standard deviation of the noise is unknown and will be
            estimated (note: this requires a prior to be given for sigma). If
            not None, this defines the standard-deviation of the data points.
            This can either be a single float, or an array with length equal
            to that for `x` and `y`.
        """

        super(GaussianCensoredLikelihood, self).__init__(x=x, y=y, func=func, sigma=sigma, **kwargs)
        self.yUL = np.array(yUL, dtype=bool)

    def log_likelihood(self):
        log_l_main = np.sum(- (self.residual[~self.yUL] / self.sigma[~self.yUL])**2 / 2 -
                       np.log(2 * np.pi * self.sigma[~self.yUL]**2) / 2)
        log_l_censored = np.sum(np.log(0.5 + 0.5*erf(self.residual[self.yUL]/(np.sqrt(2)*self.sigma[self.yUL]))))
        log_l = log_l_main + log_l_censored
        return log_l

    def __repr__(self):
        return self.__class__.__name__ + '(x={}, y={}, func={}, sigma={})' \
            .format(self.x, self.y, self.func.__name__, self.sigma)


class GeorgeCensoredLikelihood(likelihood.GeorgeLikelihood):

    def __init__(self, kernel, mean_model, t, y, yerr=1e-6,  yUL=None):
        """
            Basic Gaussian Process likelihood interface for `george' with censored data

            Parameters
            ==========
            kernel: george.kernels.Kernel
                `celerite` or `george` kernel. See the respective package documentation about the usage.
            mean_model: george.modeling.Model
                Mean model
            t: array_like
                The `times` or `x` values of the data set.
            y: array_like
                The `y` values of the data set.
            yerr: float, int, array_like, optional
                The error values on the y-values. If a single value is given, it is assumed that the value
                applies for all y-values. Default is 1e-6, effectively assuming that no y-errors are present.
            yUL: array_like (boolean)
                Flags for whether the data is left censored (upper limits).
            
            For this likelihood we assume the upper limits are not covariant with anything else, 
            so that the likelihood is easily separable.
        """
        import george
        self.yUL = np.array(yUL, dtype=bool)
        self.tUL_data = t[self.yUL]
        self.yUL_data = y[self.yUL]
        self.yerrUL_data = yerr[self.yUL]
        t = t[~self.yUL]
        y = y[~self.yUL]
        yerr = yerr[~self.yUL]
        super().__init__(kernel=kernel, mean_model=mean_model, t=t, y=y, yerr=yerr)

    def log_likelihood(self):
        """
        Calculate the log-likelihood for the Gaussian process given the current parameters.

        Returns
        =======
        float: The log-likelihood value.
        """
        for name, value in self.parameters.items():
            try:
                self.gp.set_parameter(name=name, value=value)
            except ValueError:
                raise ValueError(f"Parameter {name} not a valid parameter for the GP.")
        try:
            log_l_censored = np.sum(np.log(0.5 + 0.5*erf((self.mean_model.get_value(self.tUL_data) - self.yUL_data)/(np.sqrt(2)*self.yerrUL_data))))
            return self.gp.log_likelihood(self.y) + log_l_censored
        except Exception:
            return -np.inf

    

# tests
if __name__ == "__main__":
    test_data = np.array([1, 2, 3, 4, 5])
    result = get_credible_interval(test_data, interval=0.68)
    print(result)
    kernel = 0.2 * george.kernels.Matern32Kernel(5.0, block=(73, 230))
    from SEDPriors import SEDPriors
    priors = SEDPriors()
    george_model = LinearModel
    george_model_defaults = priors.linear_gp_defaults
    print(LinearModel(*george_model_defaults).get_value(np.array([1,2,3])))
    #exit()
    test = GeorgeCensoredLikelihood(kernel = kernel, mean_model = george_model(*george_model_defaults), \
    t = np.array([1,2,3,4]), y = np.array([1,2,3,4]), yerr = np.array([0.1]*4), yUL = [False,False,False,True])
    val = test.log_likelihood()
    print(val)