#!/usr/bin/python
import sys
import os

import numpy as np
import pandas as pd
import pickle

# import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt

rc("font", **{"family": "serif", "size": 16})  # STIXGeneral
import matplotlib.ticker as mticker

# my modules
from RaiSEDModel import RaiSEDModel
from RaiSEDGPModel import RaiSEDGPModel
from SEDPriors import SEDPriors
from Plotter import Plotter
from Fitter import Fitter
from SEDDataParser import SEDDataParser
from helper_functions import *
from auxiliary_info_functions import *


# initialise all the various classes needed
fitter = Fitter(overwrite=False)
plotter = Plotter(plotpath="output/model_plots/")
priors = SEDPriors()
parser = SEDDataParser(use_local=False)

# pick a source to fit!
src_racs_name = "RACS_2323-50A_5299"

# get the iau name for this source in the RACS LOW catalogue
src_iau_name = racs_id_to_name(src_racs_name)

# get its position
src_ra, src_dec = resolve_name_racs(src_iau_name)

# get all of the various flux density measurements associated with this source
flux_data, peak_flux_data = parser.retrieve_fluxdata_remote(
    iau_name=src_iau_name, racs_id=src_racs_name, ra=src_ra, dec=src_dec
)
# flux_data, peak_flux_data, alma_variable = parser.retrieve_fluxdata_local(iau_name = src_iau_name, racs_id = src_racs_name)

# now initialise fitter
fitter.update_data(data=flux_data, peak_data=peak_flux_data, name=src_racs_name)

# setup models depending on whether or not we require a GP (required if we have GLEAM data)
if "GLEAM" in flux_data["Survey quickname"].tolist():
    with open("data/models/gp_modelset.pkl", "rb") as f:
        model_list = pickle.load(f)
else:
    with open("data/models/modelset.pkl", "rb") as f:
        model_list = pickle.load(f)

# tell me what models we are running!
print("running RaiSED for {} using models:".format(src_racs_name))
for model in model_list:
    print(model["model_type"])

# for a simple power law - this is how we would run one model individually
# linear_gp_dict = {'has_GLEAM': True, 'model_type': 'linear_gp', 'model_func': linear_sed_func, \
# 	'plot_colour': 'darkolivegreen', 'prior_obj': priors.linear_gp_prior(), \
# 	'george_model' : LinearModel, 'george_model_defaults' : priors.linear_gp_defaults}
# result = fitter.run_single_model(**linear_gp_dict)

# do the fitting
result_array = fitter.run_all_models(model_list)
result_array, fit_params = fitter.analyse_fits(result_array)

# now get some plots!
plotter.update_data(
    data=flux_data, peak_data=peak_flux_data, name=src_racs_name, savestr_end=""
)
plotter.update_results(result_array)
plotter.plot_all_models()
plotter.plot_epoch()
plotter.plot_survey()
plotter.plot_publication()
plotter.plot_best_model()