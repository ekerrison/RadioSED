from helper_functions import *
from SEDPriors import SEDPriors
import pickle 
import os

priors = SEDPriors()

####################################
# GP models used for GLEAM data
#for a simple power law
linear_gp_dict = {'has_GLEAM': True, 'model_type': 'linear_gp', 'model_func': linear_sed_func, \
	'plot_colour': 'darkolivegreen', 'prior_obj': priors.linear_gp_prior(), \
	'george_model' : LinearModel, 'george_model_defaults' : priors.linear_gp_defaults}

#log-space parabola
orienti_gp_dict = {'has_GLEAM': True, 'model_type': 'orienti_gp', 'model_func': orienti_sed_func, \
	'plot_colour': 'b', 'prior_obj': priors.orienti_gp_prior(), \
	'george_model' : OrientiModel, 'george_model_defaults' : priors.orienti_gp_defaults}

#simple PS model
snellen_gp_dict = {'has_GLEAM': True, 'model_type': 'snellen_gp', 'model_func': snellen_sed_func, \
	'plot_colour': 'darkorange', 'prior_obj': priors.snellen_gp_prior(), \
	'george_model' : SnellenModel, 'george_model_defaults' : priors.snellen_gp_defaults}

#retriggered PS model
retrig_gp_dict = {'has_GLEAM': True, 'model_type': 'retriggered_gp', 'model_func': retriggered_sed_func, \
	'plot_colour': 'darkviolet', 'prior_obj': priors.retrig_gp_prior(), \
	'george_model' : RetriggeredModel, 'george_model_defaults' : priors.retrig_gp_defaults}

gp_model_list = [linear_gp_dict, orienti_gp_dict, snellen_gp_dict, retrig_gp_dict]

#####################################
# non-GP models used when no GLEAM data available
linear_dict = {'has_GLEAM': False, 'model_type': 'linear', 'model_func': linear_sed_func, \
	'plot_colour': 'darkolivegreen', 'prior_obj': priors.linear_prior()}

#log-space parabola
orienti_dict = {'has_GLEAM': False, 'model_type': 'orienti', 'model_func': orienti_sed_func, \
	'plot_colour': 'b', 'prior_obj': priors.orienti_prior()}

#simple PS model
snellen_dict = {'has_GLEAM': False, 'model_type': 'snellen', 'model_func': snellen_sed_func, \
	'plot_colour': 'darkorange', 'prior_obj': priors.snellen_prior()}

#retriggered PS model
retrig_dict = {'has_GLEAM': False, 'model_type': 'retriggered', 'model_func': retriggered_sed_func, \
	'plot_colour': 'darkviolet', 'prior_obj': priors.retrig_prior()}

model_list = [linear_dict, orienti_dict, snellen_dict, retrig_dict]


#####################################
# write both to a pickle file to save!
if __name__ == '__main__':
	os.makedirs('./data/models/', exist_ok = True)
	with open('./data/models/gp_modelset.pkl', 'wb+') as f:
		pickle.dump(gp_model_list, f)

	with open('./data/models/modelset.pkl', 'wb+') as f:
		pickle.dump(model_list, f)