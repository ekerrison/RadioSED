import bilby
import numpy as np
import pandas as pd
from RaiSEDModel import RaiSEDModel
from RaiSEDGPModel import RaiSEDGPModel
import helper_functions as helper


class Fitter:
    """
    Class for performing nested sampling via Bilby on the SED of a single object
    """

    def __init__(
        self,
        data: pd.DataFrame = pd.DataFrame(),
        peak_data: pd.DataFrame = pd.DataFrame(),
        model_list="all",
        name: str = None,
        lower_freq=1e7,
        upper_freq=1e11,
        output_dir="output",
        overwrite=False,
    ):
        """
        initialise default models to be fit
        """

        self.model_list = model_list
        self.maxbatch = 10
        self.n_effective = 1000
        self.overwrite = overwrite
        if data.shape[0] > 0:
            self.data = self.update_data(data, peak_data, name=name)
        self.name = name
        self.fit_min = lower_freq
        self.fit_max = upper_freq
        self.output_dir = output_dir
        return

    def update_data(self, data: pd.DataFrame, peak_data: pd.DataFrame, name: str):
        """
        Function to update the dataframe we are doing fitting on and also the source name
        """
        self.data = data
        self.peak_data = peak_data
        self.name = name

        # check that no flux values or uncertainties are NaNs!
        if self.data["Flux Density (Jy)"].isnull().values.any():
            raise ValueError(
                "Cannot fit an SED with null flux density values, please check data inputs!"
            )
        elif self.data["Uncertainty"].isnull().values.any():
            raise ValueError(
                "Cannot fit an SED with null flux uncertainty values, please check data inputs!"
            )

        # get min and max observed freq
        self.min_obs_freq = data["Frequency (Hz)"].min()
        self.max_obs_freq = data["Frequency (Hz)"].max()

        return

    def run_single_model(
        self,
        has_GLEAM=False,
        model_type=None,
        model_func=None,
        plot_colour="g",
        prior_obj=None,
        sampler_type="dynamic_dynesty",
        george_model=None,
        george_model_defaults=None,
    ):
        """
        Function to run sed fitting for a single model
        """

        if has_GLEAM:
            fit = RaiSEDGPModel(
                data=self.data,
                model_type=model_type,
                fit_min=self.fit_min,
                fit_max=self.fit_max,
                model_func=model_func,
                src_name=self.name,
                plot_colour=plot_colour,
                output_dir=self.output_dir,
                george_model=george_model,
                george_model_defaults=george_model_defaults,
            )
        else:
            fit = RaiSEDModel(
                data=self.data,
                model_type=model_type,
                fit_min=self.fit_min,
                fit_max=self.fit_max,
                model_func=model_func,
                src_name=self.name,
                plot_colour=plot_colour,
                output_dir=self.output_dir,
            )
        fit.setup_sampler(
            prior=prior_obj,
            sampler_type=sampler_type,
            maxbatch=self.maxbatch,
            use_stop=False,
            n_effective=self.n_effective,
            overwrite=self.overwrite,
        )
        fit.run_sampler()
        return fit

    def run_all_models(self, model_list: list):
        """
        Function to run fitting for a set of models, then determine which of those is the best
        fitting and return relevant parameters for the fit
        """
        # get models
        results_list = []
        for model_dict in model_list:
            results_list.append(self.run_single_model(**model_dict))

        result_array = np.asarray(results_list)

        return result_array

    def analyse_fits(self, result_array):
        """
        Function to analyse the output of all of the fitting.
        This function:
        1) determines which model is the best according to the evidence, and
            reorders the result_array so that it is in decreasing order of
            evidence
        2) Derives parameter estimates for the best fitting model

        It then returns the re-ordered result_array and the fit_params for the
        best model
        """
        # get the best model!
        result_array, log10Z_arr, bmd_array = helper.get_best_model(result_array)

        peaked_spectrum = False

        if (
            "retrig" in result_array[0].model_type
            or "orienti" in result_array[0].model_type
        ):
            peaked_spectrum = True

            # get alphas and peak freq/peak flux
            # peak_freq_interval, peak_flux_interval, trough_freq_interval, \
            # trough_flux_interval, alpha_retrig_interval, alpha_thick_interval, \
            # alpha_thin_interval, func_type
            best_model_info = helper.get_retrig_intervals(
                result_array[0],
                SED_func=result_array[0].__SED_func__,
                gp=result_array[0].gp,
                freq_min=self.lower_freq,
                freq_max=self.upper_freq,
                min_obs_freq=self.min_obs_freq,
                max_obs_freq=self.max_obs_freq,
            )

            fit_params = []
            for params in best_model_info:
                temp = template_interval()
                temp.median, temp.minus, temp.plus = params[0], params[1], params[2]
                fit_params.extend([temp])

        elif "orienti" in result_array[0].model_type:
            peaked_spectrum = True

            best_model_info = helper.get_orienti_intervals(
                result_array[0],
                SED_func=result_array[0].__SED_func__,
                gp=result_array[0].gp,
                freq_min=self.lower_freq,
                freq_max=self.upper_freq,
                min_obs_freq=self.min_obs_freq,
                max_obs_freq=self.max_obs_freq,
            )

            fit_params = []
            for params in best_model_info:
                temp = template_interval()
                temp.median, temp.minus, temp.plus = params[0], params[1], params[2]
                fit_params.extend([temp])

        elif "snellen" in result_array[0].model_type:
            peaked_spectrum = True

            fit_params = result_array[0].get_param_medians_errors()

        elif "lin" in result_array[0].model_type:
            peaked_spectrum = False
            fit_params = result_array[0].get_param_medians_errors()

        return result_array, fit_params


class TemplateInterval:
    def __init__(self):
        self.median = None
        self.plus = None
        self.minus = None
