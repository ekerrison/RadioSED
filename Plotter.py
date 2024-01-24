import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import helper_functions as helper
import pandas as pd
import numpy as np


class Plotter:
    def __init__(
        self,
        data: pd.DataFrame = None,
        peak_data: pd.DataFrame = None,
        result_array=None,
        name: str = None,
        plotpath="output/model_plots/",
        z: float = None,
        lower_freq=1e7,
        upper_freq=9e11,
        savestr_end="",
    ):
        self.result_array = self.update_results(result_array)
        if data is not None:
            self.data, self.peak_data = self.update_data(
                data, peak_data, name, savestr_end
            )
        self.plotpath = plotpath
        self.fit_min = lower_freq
        self.fit_max = upper_freq
        self.plot_curves = None

        if not os.path.isdir(os.path.join(os.getcwd(), self.plotpath)):
            os.mkdir(os.path.join(os.getcwd(), self.plotpath))
        return

    def update_results(self, result_array):
        self.result_array = result_array
        #update models
        if self.result_array is not None:
            self.plot_curves = [self.result_array[0].get_best_fit_func()]

        return

    def update_data(
        self, data: pd.DataFrame, peak_data: pd.DataFrame, name: str, savestr_end: str
    ):
        """
        Function to update the dataframe we are doing fitting on and also the source name
        """
        self.data = data
        self.name = name
        self.peak_data = peak_data
        self.savestr_end = savestr_end

        if "Upper limits" not in self.data.columns.tolist():
                self.data["Upper limits"] = False

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

        # get plotting range of data
        self.plot_freqs = (
            self.data.loc[
                (self.data["Frequency (Hz)"] <= self.fit_max)
                & (self.data["Frequency (Hz)"] >= self.fit_min),
                "Frequency (Hz)",
            ]
            / 1e6
        )
        self.plot_fluxes = self.data.loc[
            (self.data["Frequency (Hz)"] <= self.fit_max)
            & (self.data["Frequency (Hz)"] >= self.fit_min),
            "Flux Density (Jy)",
        ]
        self.plot_errs = self.data.loc[
            (self.data["Frequency (Hz)"] <= self.fit_max)
            & (self.data["Frequency (Hz)"] >= self.fit_min),
            "Uncertainty",
        ]
        self.plot_epochs = self.data.loc[
            (self.data["Frequency (Hz)"] <= self.fit_max)
            & (self.data["Frequency (Hz)"] >= self.fit_min),
            "Refcode",
        ].apply(lambda x: int(x[0:4]))

        # plot peak photometry if it exists
        if peak_data is not None:
            self.plot_peak_freqs = (
                self.peak_data.loc[
                    (self.peak_data["Frequency (Hz)"] <= self.fit_max)
                    & (self.peak_data["Frequency (Hz)"] >= self.fit_min),
                    "Frequency (Hz)",
                ]
                / 1e6
            )
            self.plot_peak_fluxes = self.peak_data.loc[
                (self.peak_data["Frequency (Hz)"] <= self.fit_max)
                & (self.peak_data["Frequency (Hz)"] >= self.fit_min),
                "Flux Density (Jy)",
            ]
            self.plot_peak_errs = self.peak_data.loc[
                (self.peak_data["Frequency (Hz)"] <= self.fit_max)
                & (self.peak_data["Frequency (Hz)"] >= self.fit_min),
                "Uncertainty",
            ]

        return

    def init_plot(self):
        """Function to create the generic plot object that is used for many different
        representations of the data"""

        fig, ax = plt.subplots(figsize=(7, 5))
        # plot very best model of each kind
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Flux Density (Jy)")
        ax.set_yscale("log")
        ax.set_xscale("log")
        # plt.grid(True)
        ax.set_xlim(0.5 * min(self.plot_freqs), 2 * max(self.plot_freqs))
        ax.set_ylim(0.5 * min(self.plot_fluxes), 5 * max(self.plot_fluxes))

        # dummy xarray for plotting models
        xx = 10 ** (
            np.linspace(np.log10(self.fit_min / 1e6), np.log10(self.fit_max / 1e6), 100)
        )

        return fig, ax, xx

    def add_data(self, fig, ax):
        """Function to add data to an initialised plot"""

        # put data on the figure
        # plotting points for peak flux densities
        if self.peak_data is not None:
            ax.errorbar(
                self.plot_peak_freqs,
                self.plot_peak_fluxes,
                yerr=self.plot_peak_errs,
                linewidth=1,
                fmt="x",
                color="darkgray",
                markersize=5,
                zorder=500,
            )

        # plotting integrated flux densities
        main_pts, main_caps, main_barlinecols = ax.errorbar(
            self.plot_freqs[self.data["Upper limits"] == False],
            self.plot_fluxes[self.data["Upper limits"] == False],
            yerr=self.plot_errs[self.data["Upper limits"] == False],
            linewidth=1,
            fmt="o",
            color="k",
            markersize=2,
            zorder=1000,
        )

        #appending upper limits
        if self.data[self.data['Upper limits'] == True].shape[0] > 0:
            lim_pts, lim_caps, lim_barlinecols = ax.errorbar(
                self.data.loc[self.data["Upper limits"] == True, 'Frequency (Hz)']/1e6,
                self.data.loc[self.data["Upper limits"] == True, 'Flux Density (Jy)'],
                yerr=self.data.loc[self.data["Upper limits"] == True, 'Flux Density (Jy)']*0.3,
                linewidth=1,
                fmt="o",
                color="dimgrey",
                markersize=2,
                zorder=1000,
                uplims=True,
            )
        else:
             lim_pts = None
        

        return fig, ax, main_pts, lim_pts

    def plot_raw_sed(self):
        """Function to plot SED data without any fitted models."""
        fig, ax, xx = self.init_plot()
        fig, ax, main_pts, lim_pts = self.add_data(fig, ax)
        plt.savefig(
            self.plotpath
            + "_".join(self.name.split(" "))
            + "{}_raw.png".format(self.savestr_end),
            bbox_inches="tight",
        )
        plt.close()
        return 

    def plot_all_models(self):
        """Function showing the best fitting realisation of each model we fit"""

        fig, ax, xx = self.init_plot()
        fig, ax, main_pts, lim_pts = self.add_data(fig, ax)

        self.plot_curves = []
        # plot all the functions!
        for idx in range(len(self.result_array)):
            if idx == 0:
                alphaplot = 1
                lw_plot = 2
            else:
                alphaplot = 0.4
                lw_plot = 0.8

            curve_to_plot = self.result_array[idx].get_best_fit_func()
            ax.plot(
                xx,
                curve_to_plot,
                c=self.result_array[idx].plot_colour,
                linestyle=self.result_array[idx].plot_linestyle,
                label=self.result_array[idx].model_type,
                alpha=alphaplot,
                lw=lw_plot,
            )
            self.plot_curves.append(curve_to_plot)

        # insert gp for best of the gp models!
        if "gp" in self.result_array[0].model_type:
            # compute it once
            # gp_noise = snellen_gp.result.posterior.iloc[-1]["white_noise:value"]
            self.result_array[0].likelihood.gp.compute(
                self.data["Frequency (Hz)"] / 1e6, self.data["Uncertainty"]
            )
            pred_mean, pred_var = self.result_array[0].likelihood.gp.predict(
                self.data["Flux Density (Jy)"], xx, return_var=True
            )
            pred_std = np.sqrt(pred_var)
            ax.plot(
                xx, pred_mean, c="m", lw=1, label="Best GP predicted mean", alpha=0.6
            )
            ax.fill_between(
                xx,
                pred_mean + pred_std,
                pred_mean - pred_std,
                color="m",
                alpha=0.3,
                edgecolor="none",
            )

            # also plot mean function!
            trend = self.result_array[0].likelihood.mean_model.get_value(xx)
            ax.plot(xx, trend, color="m", alpha=0.3, lw=5, label="Mean")

            # print('george model outputs:')
            # print(result_array[0].likelihood.gp.get_parameter_names())
            # print(result_array[0].likelihood.gp.get_parameter_vector())

            # print('parameters in best fit plot:')
            # print(result_array[0].fit_params_func)

        ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.15))

        # add text with BIC for all models
        # we want to do models ordered by complexity (result.get_md()) using model_type for the label
        dims = np.asarray([x.get_md() for x in self.result_array])
        dim_order = np.argsort(dims)
        y_pos = 1.03
        if "GLEAM" in self.data["Survey quickname"].tolist():
            for idx in dim_order:
                y_pos -= 0.1
                ax.text(
                    0.85,
                    y_pos,
                    "{} ".format(
                        self.result_array[idx].model_type.split("_")[0].title()
                    )
                    + "log$_{10}$ Z"
                    + ":\nGP:{:.2f}".format(self.result_array[idx].get_log10z()[0]),
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    color="k",
                    fontsize=10,
                )

            # update these to be more generic and still always ordered by number of params!
            # ax.text(0.85, 0.93, 'Linear log$_{10}$ Z' + ':\nGP:{:.2f}'.format(linear_gp.get_log10z()[0]), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, color = 'k', fontsize = 10)
            # ax.text(0.85, 0.83, 'Orienti log$_{10}$ Z' + ':\nGP:{:.2f}'.format(orienti_gp.get_log10z()[0]), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, color = 'k', fontsize = 10)
            # ax.text(0.85, 0.73, 'Snellen log$_{10}$ Z' + ':\nGP:{:.2f}'.format(snellen_gp.get_log10z()[0]), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, color = 'k', fontsize = 10)
            # ax.text(0.85, 0.63, 'Retriggered log$_{10}$ Z' + ':\nGP:{:.2f}'.format(retrig_gp.get_log10z()[0]), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, color = 'k', fontsize = 10)
            ax.text(
                0.85,
                y_pos - 0.08,
                "Best Fit: {}".format(
                    " ".join(self.result_array[0].model_type.split("_"))
                ),
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                color="k",
                fontsize=10,
                weight="bold",
            )
            ax.text(
                0.85,
                y_pos - 0.13,
                "B$_{1,2}$"
                + ": {:.3f}".format(
                    self.result_array[0].get_log10z()[0]
                    - self.result_array[1].get_log10z()[0]
                ),
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                color="k",
                fontsize=10,
                weight="bold",
            )

        else:
            for idx in dim_order:
                y_pos -= 0.1
                ax.text(
                    0.85,
                    y_pos,
                    "{} ".format(
                        self.result_array[idx].model_type.split("_")[0].title()
                    )
                    + "log$_{10}$ Z"
                    + ":\nNS:{:.2f}".format(self.result_array[idx].get_log10z()[0]),
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    color="k",
                    fontsize=10,
                )
            # ax.text(0.85, 0.93, 'Linear log$_{10}$ Z' + ':\nNS:{:.2f}'.format(linear_nested.get_log10z()[0]), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, color = 'k', fontsize = 10)
            # ax.text(0.85, 0.83, 'Orienti log$_{10}$ Z' + ':\nNS:{:.2f}'.format(orienti_nested.get_log10z()[0]), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, color = 'k', fontsize = 10)
            # ax.text(0.85, 0.73, 'Snellen log$_{10}$ Z' + ':\nNS:{:.2f}'.format(snellen_nested.get_log10z()[0]), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, color = 'k', fontsize = 10)
            # ax.text(0.85, 0.63, 'Retriggered log$_{10}$ Z' + ':\nNS:{:.2f}'.format(retrig_nested.get_log10z()[0]), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, color = 'k', fontsize = 10)
            ax.text(
                0.85,
                y_pos - 0.08,
                "Best Fit: {}".format(
                    " ".join(self.result_array[0].model_type.split("_"))
                ),
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                color="k",
                fontsize=10,
                weight="bold",
            )
            ax.text(
                0.85,
                y_pos - 0.13,
                "B$_{1,2}$"
                + ": {:.3f}".format(
                    self.result_array[0].get_log10z()[0]
                    - self.result_array[1].get_log10z()[0]
                ),
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                color="k",
                fontsize=10,
                weight="bold",
            )

        plt.title(
            self.name
        )  # title string used to be IAU name / racs comps {n} / peaked : {Y/N}

        plt.savefig(
            self.plotpath
            + "_".join(self.name.split(" "))
            + "{}.png".format(self.savestr_end),
            bbox_inches="tight",
        )
        plt.close()
        return

    def plot_model_range(self, throw_figure=False, model_idx=0, fig=None, ax=None):
        """Function showing the best fitting model for the model type with the largest Bayesian
        evidence, as well as draws from the posterior to get a feel for the range of acceptable models
        """
        # plot array of best fitting functions
        if fig is None and ax is None:
            fig, ax, xx = self.init_plot()
        else:
            temp1, temp2, xx = self.init_plot()
            plt.close(temp1)

        fig, ax, main_pts, lim_pts = self.add_data(fig, ax)

        if self.plot_curves is None:
            self.plot_curves = [self.result_array[model_idx].get_best_fit_func()]

        # plot best fit
        c_main = ["darkred", "darkblue", "darkgreen", "darkviolet", "darkorange"]
        ax.plot(
            xx,
            self.plot_curves[model_idx],
            c=c_main[model_idx],
            alpha=1,
            label="Best model",
        )

        # plot range
        c_range = ["firebrick", "royalblue", "forestgreen", "mediumpurple", "orange"]
        fit_range = self.result_array[model_idx].get_fit_range_funcs()
        fit_range = fit_range.T
        xarr = np.array([xx] * fit_range.shape[1]).T
        ax.plot(xarr, fit_range, lw=1, color=c_range[model_idx], alpha=0.15)

        if throw_figure:
            return fig, ax, main_pts

        # get param strings for plotting!
        str_toplot_list = []
        str_toplot_list = helper.get_param_strs_toplot(self.result_array[0])
        len_str_toplot_list = len(str_toplot_list)

        # add main labels to plot
        if len_str_toplot_list >= 4:
            main_label_x = 0.7
        else:
            main_label_x = 0.75
        ax.text(
            main_label_x,
            0.95,
            "Best Fit BF: {}".format(
                " ".join(self.result_array[0].model_type.split("_"))
            ),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="k",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            main_label_x,
            0.90,
            "log10 Evidence: {:.3f}".format(self.result_array[0].get_log10z()[0]),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="k",
            fontsize=10,
            weight="bold",
        )

        # and param strings to the plot
        for str_plot_idx in range(len_str_toplot_list):
            plotstr = str_toplot_list[str_plot_idx]
            if len_str_toplot_list >= 4:
                ax.text(
                    0.55 + 0.3 * (str_plot_idx % 2),
                    0.83 - int(str_plot_idx / 2) * 0.06,
                    plotstr,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    color="k",
                    fontsize=10,
                )
            else:
                ax.text(
                    0.75,
                    0.83 - str_plot_idx * 0.06,
                    plotstr,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    color="k",
                    fontsize=10,
                )

        fig.savefig(
            self.plotpath
            + "_".join(self.name.split(" "))
            + "_model{}_{}.png".format(model_idx, self.savestr_end),
            bbox_inches="tight",
        )
        plt.close("all")
        return

    def plot_best_model(self, throw_figure=False):
        """Function to plot the best fitting model as determined by the Bayes factor
        (Bayesian Evidence). This plots both the best fitting realisation, as well as 
        several addtional realisations from the posterior. Text is added to provide
        useful information about the model, such as the parameter estimates."""
        fig, ax, main_pts = self.plot_model_range(throw_figure=True, model_idx=0)

        if throw_figure:
            return fig, ax, main_pts

        # get param strings for plotting!
        str_toplot_list = []
        str_toplot_list = helper.get_param_strs_toplot(self.result_array[0])
        len_str_toplot_list = len(str_toplot_list)

        # add main labels to plot
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        if len_str_toplot_list >= 4:
            main_label_x = 0.7
        else:
            main_label_x = 0.75
        ax.text(
            main_label_x,
            0.95,
            "Best Fit BF: {}".format(
                " ".join(self.result_array[0].model_type.split("_"))
            ),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="k",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            main_label_x,
            0.90,
            "log10 Evidence: {:.3f}".format(self.result_array[0].get_log10z()[0]),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="k",
            fontsize=10,
            weight="bold",
        )

        # and add them to the plot
        for str_plot_idx in range(len_str_toplot_list):
            plotstr = str_toplot_list[str_plot_idx]
            if len_str_toplot_list >= 4:
                ax.text(
                    0.55 + 0.3 * (str_plot_idx % 2),
                    0.83 - int(str_plot_idx / 2) * 0.06,
                    plotstr,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    color="k",
                    fontsize=10,
                )
            else:
                ax.text(
                    0.75,
                    0.83 - str_plot_idx * 0.06,
                    plotstr,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    color="k",
                    fontsize=10,
                )

        plt.savefig(
            self.plotpath
            + "_".join(self.name.split(" "))
            + "_bestfit{}.png".format(self.savestr_end),
            bbox_inches="tight",
        )
        plt.close()
        return

    def plot_epoch(self):
        """Function to plot pooints coloured by epoch, based on the date captured
        as part of the Bibcode for each survey."""

        fig, ax, main_pts = self.plot_best_model(throw_figure=True)

        # get rid of those standard black points from the plot!
        main_pts.remove()

        # epoch
        ax.errorbar(
            self.plot_freqs,
            self.plot_fluxes,
            yerr=self.plot_errs,
            linewidth=1,
            fmt="none",
            color="k",
            markersize=2,
        )
        cdata = ax.scatter(
            self.plot_freqs,
            self.plot_fluxes,
            s=20,
            c=self.plot_epochs,
            cmap="viridis",
            zorder=1000,
        )
        fig.colorbar(cdata, label="epoch")

        plt.savefig(
            self.plotpath
            + "_".join(self.name.split(" "))
            + "_bestfit{}_epoch.png".format(self.savestr_end),
            bbox_inches="tight",
        )
        plt.close()
        return

    def plot_survey(self):
        """Function to plot points coloured by survey, using the survey names within
        the 'Survey quicknames' column of the flux data table."""

        fig, ax, main_pts = self.plot_best_model(throw_figure=True)

        # get rid of those standard black points from the plot!
        main_pts.remove()

        # error bars
        plt.errorbar(
            self.plot_freqs,
            self.plot_fluxes,
            yerr=self.plot_errs,
            linewidth=1,
            fmt="none",
            color="k",
            markersize=2,
        )
        # loop through surveys!
        colour_list = [
            "royalblue",
            "navy",
            "deepskyblue",
            "teal",
            "steelblue",
            "lightsteelblue",
            "darkslategray",
            "blue",
            "r",
            "g",
            "c",
            "m",
        ]
        markerlist = ["X", "o", "+", "^", "*", "1", "P", "d", "X", "o", "+"]
        c_idx = 0
        for surv_name in self.data["Survey quickname"].unique().tolist():
            plot_freqs_s = (
                self.data.loc[
                    self.data["Survey quickname"] == surv_name, "Frequency (Hz)"
                ]
                / 1e6
            )
            plot_fluxes_s = self.data.loc[
                self.data["Survey quickname"] == surv_name, "Flux Density (Jy)"
            ]
            ax.scatter(
                plot_freqs_s,
                plot_fluxes_s,
                s=30,
                c=colour_list[c_idx % len(colour_list)],
                marker=markerlist[c_idx % len(markerlist)],
                zorder=1000,
                label=surv_name,
            )
            c_idx += 1
        plt.legend(fontsize=10)

        plt.savefig(
            self.plotpath
            + "_".join(self.name.split(" "))
            + "_bestfit{}_survey.png".format(self.savestr_end),
            bbox_inches="tight",
        )
        # plt.show()
        plt.close()
        return

    def plot_publication(self, name = None):
        """Functionally the same as plot_best_model but without any text on the 
        plot except for the source name, and with ticks inside axis boundaries."""
        # plot array of best fitting functions

        if name is None:
            name = self.name

        fig, ax, main_pts = self.plot_best_model(throw_figure=True)
        # put tick params inside
        ax.tick_params(axis="y", direction="in", which="both", length=3, right=True)
        ax.tick_params(axis="x", direction="in", which="both", length=3, top=True)
        # and make them not scientific
        #ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1g"))
        # plt.legend(loc = 'upper center', ncol = 3, bbox_to_anchor=(0.5, -0.15))
        # plt.grid(True)
        # plt.title(obj_name)
        ax.text(
            0.7,
            0.92,
            name,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        plt.savefig(
            self.plotpath
            + "_".join(name.split(" "))
            + "_bestfit{}_pub.pdf".format(self.savestr_end),
            bbox_inches="tight",
        )
        # plt.show()
        plt.close()
        return

    def plot_outofbounds(self):
        """currently just used for plotting ALMA data not used in the fit"""
        # TODO: incorporate this@
        return

    def plot_restframe(self, z=0):
        """Plot the SED in the rest frame if we know its redshift"""
        if z is None or z <= 0:
            return

        fig, ax, xx = self.init_plot()

        # shift frequencies into rest frame!
        rest_plot_freqs = (1 + z) * self.plot_freqs
        if self.peak_data is not None:
            rest_plot_peak_freqs = (1 + z) * self.plot_peak_freqs
        xx = (1 + z) * xx

        # plot peak data points
        if self.peak_data is not None:
            ax.errorbar(
                rest_plot_peak_freqs,
                self.plot_peak_fluxes,
                yerr=self.plot_peak_errs,
                linewidth=1,
                fmt="x",
                color="darkgray",
                markersize=5,
            )

        # plot all flux data points
        ax.errorbar(
            rest_plot_freqs,
            self.plot_fluxes,
            yerr=self.plot_errs,
            linewidth=1,
            fmt="o",
            color="k",
            markersize=2,
        )

        # plot best fit
        if self.plot_curves is None:
            self.plot_curves = [self.result_array[0].get_best_fit_func()]
        ax.plot(xx, plot_curves[0], c="darkred", alpha=1, label="Best model")

        # plot range
        fit_range = self.result_array[0].get_fit_range_funcs()
        fit_range = fit_range.T
        xarr = np.array([xx] * fit_range.shape[1]).T
        ax.plot(xarr, fit_range, lw=1, color="firebrick", alpha=0.15)

        # put tick params inside
        ax.tick_params(axis="y", direction="in", which="both", length=3, right=True)
        ax.tick_params(axis="x", direction="in", which="both", length=3, top=True)
        # and make them not scientific
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        # plt.legend(loc = 'upper center', ncol = 3, bbox_to_anchor=(0.5, -0.15))
        # plt.grid(True)
        # plt.title(obj_name)
        ax.text(
            0.7,
            0.92,
            obj_name,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.xlim(0.5 * min(rest_plot_freqs), 2 * max(rest_plot_freqs))

        plt.savefig(
            self.plotpath
            + "_".join(self.name.split(" "))
            + "_bestfit{}_restframe.pdf".format(self.savestr_end),
            bbox_inches="tight",
        )
        plt.close()
        return

    def plot_all(self):
        """Essentially a wrapper function that calls all of the other plotting
        functions in this class including:
        plot_all_models()
        plot_best_model()
        plot_epoch()
        plot_survey()
        plot_publication()
        plot_outofbounds()"""
        self.plot_all_models()
        self.plot_best_model()
        self.plot_epoch()
        self.plot_survey()
        self.plot_publication()
        self.plot_outofbounds()
        if self.z is not None:
            self.plot_restframe()
        return
