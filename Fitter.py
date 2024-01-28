import bilby
import numpy as np
import pandas as pd
from RadioSEDModel import RadioSEDModel
from RadioSEDGPModel import RadioSEDGPModel
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
        upper_freq=9e11,
        output_dir="output",
        overwrite=False,
        use_nestcheck=False,
        savestr_end="",
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
        self.use_nestcheck = use_nestcheck
        self.savestr_end = savestr_end
        return

    def update_data(self, data: pd.DataFrame, peak_data: pd.DataFrame, name: str, savestr_end: str = ""):
        """
        Function to update the dataframe we are doing fitting on and also the source name
        """
        self.data = data
        self.peak_data = peak_data
        self.name = name
        self.savestr_end = savestr_end

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
            fit = RadioSEDGPModel(
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
                use_nestcheck=self.use_nestcheck,
                savestr_end=self.savestr_end,
            )
        else:
            fit = RadioSEDModel(
                data=self.data,
                model_type=model_type,
                fit_min=self.fit_min,
                fit_max=self.fit_max,
                model_func=model_func,
                src_name=self.name,
                plot_colour=plot_colour,
                output_dir=self.output_dir,
                use_nestcheck=self.use_nestcheck,
                savestr_end=self.savestr_end,
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
        best model, as well as the log10 evidence of each model
        """
        # get the best model!
        result_array, log10Z_arr, bmd_array = helper.get_best_model(result_array)

        peaked_spectrum = False

        if (
            "retrig" in result_array[0].model_type
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
                freq_min=self.fit_min,
                freq_max=self.fit_max,
                min_obs_freq=self.min_obs_freq,
                max_obs_freq=self.max_obs_freq,
            )

            fit_params = []
            fit_param_names = ['peak_freq', 'peak_flux', 'trough_freq', 'trough_flux',\
                'alpha_retrig', 'alpha_thick', 'alpha_thin']
            for param_idx in range(len(fit_param_names)):
                params = best_model_info[param_idx]
                temp = TemplateInterval()
                temp.name = fit_param_names[param_idx]
                temp.median, temp.minus, temp.plus = params[0], params[1], params[2]
                fit_params.extend([temp])

        elif "orienti" in result_array[0].model_type:
            peaked_spectrum = True

            best_model_info = helper.get_orienti_intervals(
                result_array[0],
                SED_func=result_array[0].__SED_func__,
                gp=result_array[0].gp,
                freq_min=self.fit_min,
                freq_max=self.fit_max,
                min_obs_freq=self.min_obs_freq,
                max_obs_freq=self.max_obs_freq,
            )

            fit_params = []
            fit_param_names = ['peak_freq', 'peak_flux', 'alpha_thick', 'alpha_thin']
            for param_idx in range(len(fit_param_names)):
                params = best_model_info[param_idx]
                temp = TemplateInterval()
                temp.name = fit_param_names[param_idx]
                temp.median, temp.minus, temp.plus = params[0], params[1], params[2]
                fit_params.extend([temp])

        elif "snellen" in result_array[0].model_type:
            peaked_spectrum = True

            fit_params_bilby = result_array[0].get_param_medians_errors()

            fit_params = []
            fit_param_names = ['peak_freq', 'peak_flux', 'alpha_thick', 'alpha_thin', 'Const', 'M00']
            for param_idx in range(len(fit_params_bilby)):
                params = fit_params_bilby[param_idx]
                temp = TemplateInterval()
                temp.name = fit_param_names[param_idx]
                temp.median, temp.minus, temp.plus = params.median, params.plus, params.minus
                fit_params.extend([temp])

        elif "lin" in result_array[0].model_type:
            peaked_spectrum = False
            
            fit_params_bilby = result_array[0].get_param_medians_errors()

            fit_params = []
            fit_param_names = ['S_norm', 'alpha', 'Const', 'M00']
            for param_idx in range(len(fit_params_bilby)):
                params = fit_params_bilby[param_idx]
                temp = TemplateInterval()
                temp.name = fit_param_names[param_idx]
                temp.median, temp.minus, temp.plus = params.median, params.plus, params.minus
                fit_params.extend([temp])

        return result_array, fit_params, log10Z_arr


class TemplateInterval:
    def __init__(self):
        self.median = None
        self.plus = None
        self.minus = None
        self.name = None

    def __str__(self):
        return "{}: {} +/- {} / {}".format(self.name, self.median, self.plus, self.minus)