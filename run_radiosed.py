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
from RadioSEDModel import RadioSEDModel
from RadioSEDGPModel import RadioSEDGPModel
from SEDPriors import SEDPriors
from Plotter import Plotter
from Fitter import Fitter
from SEDDataParser import SEDDataParser
from helper_functions import *
from auxiliary_info_functions import *

#make necessary directories if they do not exist
print('output directory exists: ', os.path.isdir(os.path.join(os.getcwd(), 'output')))
if not os.path.isdir(os.path.join(os.getcwd(), 'output')):
    os.mkdir(os.path.join(os.getcwd(), 'output'))

# initialise all the various classes needed
#NOTE nestcheck must be installed directly from the github repo, as 
#changes have been made since the last release to ensure it works with dynamic nested
#sampling from dynesty using batch sampling (see: https://github.com/ejhigson/nestcheck/pull/9)
fitter = Fitter(overwrite=False, use_nestcheck=False)
plotter = Plotter(plotpath="output/model_plots/")
priors = SEDPriors()

# set use_local=True if you want to use the crossmatches that come pre-matched with RadioSED
# this is the recommended setting for if you want to fit many sources (>20 or so) as it removes
# the overheads from querying Vizier repeatedly. If you want to run RadioSED on a cluster/HPC
# then you should probably set use_local=True, as repeated queries to Vizier may slow down 
# traffic
parser = SEDDataParser(use_local=False)

# pick a source to fit - we want the closest RACS source to this one
src_iau_name_auto, ra, dec, separation, racs_id = resolve_name_generic('J135753+004633')

# or we know it's RACS ID
src_racs_name = "RACS_2323-50A_5299"

# get the iau name for this source in the RACS LOW catalogue
#src_iau_name = racs_id_to_name(src_racs_name)

#check they are the same in this case
print('names match: {}'.format(src_iau_name_auto == src_iau_name_auto))

# get its position
src_ra, src_dec = resolve_name_racs(src_iau_name_auto)

# get all of the various flux density measurements associated with this source
flux_data, peak_flux_data, alma_variable = parser.retrieve_fluxdata_remote(
    iau_name=src_iau_name_auto, racs_id=racs_id, ra=ra, dec=dec
)

#the same but using the local crossmatches
# flux_data, peak_flux_data, alma_variable = parser.retrieve_fluxdata_local(iau_name = src_iau_name, racs_id = src_racs_name)

# now initialise fitter
fitter.update_data(data=flux_data, peak_data=peak_flux_data, name=src_iau_name_auto)

# setup models depending on whether or not we require a GP (required if we have GLEAM data)
if "GLEAM" in flux_data["Survey quickname"].tolist():
    with open("data/models/gp_modelset.pkl", "rb") as f:
        model_list = pickle.load(f)
else:
    with open("data/models/modelset.pkl", "rb") as f:
        model_list = pickle.load(f)

# tell me what models we are running!
print("running RaiSED for {} using models:".format(src_iau_name_auto))
for model in model_list:
    print(model["model_type"])

# for a simple power law - this is how we would run one model individually
# linear_gp_dict = {'has_GLEAM': True, 'model_type': 'linear_gp', 'model_func': linear_sed_func, \
# 	'plot_colour': 'darkolivegreen', 'prior_obj': priors.linear_gp_prior(), \
# 	'george_model' : LinearModel, 'george_model_defaults' : priors.linear_gp_defaults}
# result = fitter.run_single_model(**linear_gp_dict)

# do the fitting
result_array = fitter.run_all_models(model_list)
result_array, fit_params, log10Z_arr = fitter.analyse_fits(result_array)

#print out the parameters given by fitting two different ways
print('Best fit model type: ', result_array[0].model_type)
print('Number of parameters in best fit: ', len(fit_params))
print(fit_params[0])
print(fit_params[0].name, fit_params[0].median, fit_params[0].plus, fit_params[0].minus)

# now get some plots!
plotter.update_data(
    data=flux_data, peak_data=peak_flux_data, name=src_iau_name_auto, savestr_end=""
)
plotter.update_results(result_array)
plotter.plot_all_models()
plotter.plot_epoch()
plotter.plot_survey()
plotter.plot_publication()
plotter.plot_best_model()
