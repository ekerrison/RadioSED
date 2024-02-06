#!/usr/bin/python
import sys
import os

import numpy as np
import pandas as pd
import pickle

# my modules
# my modules
from RadioSEDModel import RadioSEDModel
from RadioSEDGPModel import RadioSEDGPModel
from SEDPriors import SEDPriors
from Plotter import Plotter
from Fitter import Fitter
from SEDDataParser import SEDDataParser
from auxiliary_info_functions import AuxInfo

#argparse
import argparse

parser = argparse.ArgumentParser(prog='RadioSED',
    description='Run RadioSED from the command line.',
    epilog='Please refer to the README for more information.')
parser.add_argument('-n', '--name', help='''The IAU designation of the source to be fit, in the format
                    Jhhmmss.ss+ddmmss.ss''')
parser.add_argument('-m', '--ned_name', help='''The name of the source to be fit as recognised by the
                    NASA Extragalactic master_databse (NED)''')
parser.add_argument('-i', '--id', help='''The RACS-low ID of the source to be fit''')
parser.add_argument('-x', '--position', help='''The RA and Dec of the source to be fit, in decimal
                    degrees''', nargs=2, type=float)
parser.add_argument('-f', '--file', help='''A file containing a list of sources to
                    be fit. Must be parsable as a table with at least one of 
                    ['IAU_designation', 'NED_name', 'ra', 'dec'] columns (or two in the case
                    of coordinates)''')
parser.add_argument('-p', '--plot', help='''Make plots of fits.''', action='store_true')
parser.add_argument('-o', '--overwrite', help='''Overwrite any previous fit and saved master_data files.
                If False, then any pre-existing bilby files for this source will be loaded in, 
                fitting will be resumed if it was interrupted, and only plots will be regenerated.''',
                action='store_true')
parser.add_argument('-c', '--custom_data_file', help='''Use custom flux density master_data
                    at the location specified for a given source.''')
parser.add_argument('-w', '--write_output', help='''Write numerical output to file at the specified location.
                    Default is to write to summary_data.csv under output/data/.''', nargs='?', 
                    const= '../output/data/summary_data.csv', type=str)
parser.add_argument('--max_freq', help='''Change the maximum frequency to which RadioSED will
                    fit master_data (in Hz).''', default=5e11)
parser.add_argument('-u', '--use_local', help='''Perform fitting using locally
                    stored radio catalogue crossmatches.''', action='store_true')

args = parser.parse_args()

#columns for writing to file
# get the info that is going to be saved to file
col_list = ['obj_name', 'iau_name', 'ra', 'dec', 'peaked_spectrum', 'retrig_spectrum', 'racs_extended', 'n_gaus_racs', 'n_flux', \
 'n_flux_new', 'new_only', 'gleam_blended', 'gleam_fluxratio', 'at20g_compact', 'at20g_visibility', \
 'best_model', 'best_model_log10Z', 'linear_nested_log10z', \
 'linear_gp_log10z', 'orienti_nested_log10z', 'orienti_gp_log10z', 'snellen_nested_log10z', \
 'snellen_gp_log10z', 'retrig_nested_log10z', 'retrig_gp_log10z', \
 'peak_flux', 'peak_flux_m', 'peak_flux_p', 'peak_freq', 'peak_freq_m', 'peak_freq_p', \
 'alpha_thick', 'alpha_thick_m', 'alpha_thick_p', 'alpha_thin', 'alpha_thin_m', 'alpha_thin_p', \
 'alpha_retrig', 'alpha_retrig_m', 'alpha_retrig_p', 'trough_flux', 'trough_flux_m', \
 'trough_flux_p', 'trough_freq', 'trough_freq_m', 'trough_freq_p', 'peak_outofrange', 'trough_outofrange', \
 'snorm', 'snorm_m', 'snorm_p']

#if we are writing to file, make sure the folder exists
if args.write_output is not None:
    fname = args.write_output
    if not os.path.dirname(args.write_output):
        os.makedirs(args.write_output, exist_ok=True)

    if not os.path.isfile(args.write_output):
        with open(args.write_output, 'w+') as f:
            for x in col_list:
                f.write(x + ',')
            f.write('\n')
    
    master_data = pd.read_csv(args.write_output, nrows=1)
    #make a dummy idx for table assignment
    master_data['obj_name'] = master_data['obj_name'].astype(str)
    master_data['iau_name'] = master_data['iau_name'].astype(str)
    master_data['peaked_spectrum'] = master_data['peaked_spectrum'].astype(bool)
    master_data['best_model'] = master_data['best_model'].astype(str)
    master_data['gleam_blended'] = master_data['gleam_blended'].astype(bool)
    master_data['new_only'] = master_data['new_only'].astype(bool)
    dummy_idx = 0
    count = 0


## run RadioSED!
# initialise all the various classes needed
#NOTE nestcheck must be installed directly from the github repo, as 
#changes have been made since the last release to ensure it works with dynamic nested
#sampling from dynesty using batch sampling (see: https://github.com/ejhigson/nestcheck/pull/9)
fitter = Fitter(overwrite=args.overwrite, use_nestcheck=False, upper_freq=args.max_freq)
plotter = Plotter(plotpath="../output/model_plots/", upper_freq=args.max_freq)
priors = SEDPriors()
info = AuxInfo()

# set use_local=True if you want to use the crossmatches that come pre-matched with RadioSED
# this is the recommended setting for if you want to fit many sources (>20 or so) as it removes
# the overheads from querying Vizier repeatedly. If you want to run RadioSED on a cluster/HPC
# then you should probably set use_local=True, as repeated queries to Vizier may slow down 
# traffic
parser = SEDDataParser(use_local=args.use_local)

#if no file parsed we are doing a single object
if args.file is None:
    #get the racs info for this source
    if args.name is not None:
        src_iau_name, src_ra, src_dec, separation, racs_id = info.resolve_name_generic(iau_name = args.name)

    elif args.ned_name is not None:
        racs_name, ra, dec = info.find_racs_src(ned_name = args.ned_name)
        src_iau_name, src_ra, src_dec, separation, racs_id = info.resolve_name_generic(iau_name = racs_name)

    elif args.position is not None:
        racs_name, ra, dec = info.find_racs_src(ra = position[0], dec = position[1])
        src_iau_name, src_ra, src_dec, separation, racs_id = info.resolve_name_generic(iau_name = racs_name)

    elif args.id is not None:
        src_iau_name = info.racs_id_to_name(racs_id = args.id)
        src_iau_name, src_ra, src_dec, separation, racs_id = info.resolve_name_generic(iau_name =src_iau_name)


    #get the flux master_data
    if args.use_local:
        flux_data, peak_flux_data, alma_variable, racs_id = parser.retrieve_fluxdata_local(racs_id = racs_id)
    elif not args.use_local and args.custom_data_file is None:
        flux_data, peak_flux_data, alma_variable = parser.retrieve_fluxdata_remote(iau_name = src_iau_name,
        racs_id = racs_id, ra=src_ra, dec=src_dec)
    else:
        flux_data = pd.read_csv(args.custom_data_file)
        peak_flux_data = None


    #other useful diagnostics
    # get auxiliary info about the source compactness and possible blending
    racs_n_gaus, racs_fluxratio = info.check_racs_compactness(src_name = src_iau_name) 
    gleam_blending_flag, num_gleam_neighbours = info.check_confusion(src_name = src_iau_name)
    gleam_fluxratio, gleam_sep = info.check_gleam_compactness(src_name = src_iau_name)
    racs_n_gaus, racs_fluxratio = info.check_racs_compactness(src_name = src_iau_name) 
    at20g_compactness, at20g_visibility, at20g_sep = info.check_at20g_compactness(src_name = src_iau_name)

    #remove bottom GLEAM bands if there is likely confusion
    if gleam_blending_flag == True:
        flux_data = flux_data[flux_data['Frequency (Hz)'] > 1e8]


    # now initialise fitter
    fitter.update_data(data=flux_data, peak_data=peak_flux_data, name=src_iau_name)

    # setup models depending on whether or not we require a GP (required if we have GLEAM master_data)
    if "GLEAM" in flux_data["Survey quickname"].tolist():
        with open("data/models/gp_modelset.pkl", "rb") as f:
            model_list = pickle.load(f)
    else:
        with open("data/models/modelset.pkl", "rb") as f:
            model_list = pickle.load(f)

    # tell me what models we are running!
    print("running RaiSED for {} using models:".format(src_iau_name))
    for model in model_list:
        print(model["model_type"])

    # do the fitting
    result_array = fitter.run_all_models(model_list)
    result_array, fit_params, log10z_arr = fitter.analyse_fits(result_array)
    model_type_arr = [result_array[x].model_type for x in range(len(result_array))]
    fit_param_names = [fit_params[x].name for x in range(len(fit_params))]

    #print out the parameters given by fitting two different ways
    print('Best fit model type: ', result_array[0].model_type)

    if args.plot:
        # now get some plots!
        plotter.update_data(
            data=flux_data, peak_data=peak_flux_data, name=src_iau_name, savestr_end=""
        )
        plotter.update_results(result_array)
        plotter.plot_all_models()
        plotter.plot_epoch()
        plotter.plot_survey()
        plotter.plot_publication()
        plotter.plot_best_model()

    #if we are writing to output, collect in a master_dataFrame and save
    if args.write_output:
        master_data.loc[dummy_idx, 'obj_name'] = racs_id
        master_data.loc[dummy_idx, 'iau_name'] = src_iau_name
        master_data.loc[dummy_idx, 'ra'] = src_ra
        master_data.loc[dummy_idx, 'dec'] = src_dec
        master_data.loc[dummy_idx, 'n_flux'] = flux_data.shape[0]
        master_data.loc[dummy_idx, 'peaked_spectrum'] = 'lin' not in result_array[0].model_type
        master_data.loc[dummy_idx, 'retrig_spectrum'] = 'retrig' in result_array[0].model_type
        master_data.loc[dummy_idx, 'racs_extended'] = racs_fluxratio
        master_data.loc[dummy_idx, 'n_gaus_racs'] =  racs_n_gaus
        master_data.loc[dummy_idx, 'n_flux'] = flux_data.shape[0]
        master_data.loc[dummy_idx, 'n_flux_new'] = ''
        master_data.loc[dummy_idx, 'new_only'] = False


        master_data.loc[dummy_idx, 'gleam_blended'] =  gleam_blending_flag
        master_data.loc[dummy_idx, 'gleam_fluxratio'] = gleam_fluxratio
        master_data.loc[dummy_idx, 'at20g_compact'] =  at20g_compactness
        master_data.loc[dummy_idx, 'at20g_visibility'] =  at20g_visibility

        #add racs flux
        master_data.loc[dummy_idx, 'racs_low_flux'] = flux_data.loc[flux_data['Survey quickname'] == 'RACS', 'Flux Density (Jy)'].values[0] 
        master_data.loc[dummy_idx, 'racs_low_flux_err'] = flux_data.loc[flux_data['Survey quickname'] == 'RACS', 'Uncertainty'].values[0]

        master_data.loc[dummy_idx, 'best_model'] = result_array[0].model_type
        master_data.loc[dummy_idx, 'best_model_log10Z'] = log10z_arr[0]
        if "GLEAM" in flux_data["Survey quickname"].tolist():
            master_data.loc[dummy_idx, 'linear_gp_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'lin' in s][0]]
            master_data.loc[dummy_idx, 'orienti_gp_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'orient' in s][0]]
            master_data.loc[dummy_idx, 'snellen_gp_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'snellen' in s][0]]
            master_data.loc[dummy_idx, 'retrig_gp_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'retrig' in s][0]]

            master_data.loc[dummy_idx, 'linear_nested_log10z'] = np.nan
            master_data.loc[dummy_idx, 'orienti_nested_log10z'] = np.nan
            master_data.loc[dummy_idx, 'snellen_nested_log10z'] = np.nan
            master_data.loc[dummy_idx, 'retrig_nested_log10z'] = np.nan
        
        else:
            master_data.loc[dummy_idx, 'linear_nested_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'lin' in s][0]]
            master_data.loc[dummy_idx, 'orienti_nested_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'orient' in s][0]]
            master_data.loc[dummy_idx, 'snellen_nested_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'snell' in s][0]]
            master_data.loc[dummy_idx, 'retrig_nested_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'retrig' in s][0]]
            
            master_data.loc[dummy_idx, 'linear_gp_log10z'] = np.nan
            master_data.loc[dummy_idx, 'orienti_gp_log10z'] = np.nan
            master_data.loc[dummy_idx, 'snellen_gp_log10z'] = np.nan
            master_data.loc[dummy_idx, 'retrig_gp_log10z'] = np.nan


        if 'lin' not in result_array[0].model_type:
            master_data.loc[dummy_idx, 'peak_flux'] = fit_params[fit_param_names.index('peak_flux')].median
            master_data.loc[dummy_idx, 'peak_flux_m'] = fit_params[fit_param_names.index('peak_flux')].minus
            master_data.loc[dummy_idx, 'peak_flux_p'] = fit_params[fit_param_names.index('peak_flux')].plus
            master_data.loc[dummy_idx, 'peak_freq'] = fit_params[fit_param_names.index('peak_freq')].median
            master_data.loc[dummy_idx, 'peak_freq_m'] = fit_params[fit_param_names.index('peak_freq')].minus
            master_data.loc[dummy_idx, 'peak_freq_p'] = fit_params[fit_param_names.index('peak_freq')].plus
            master_data.loc[dummy_idx, 'alpha_thick'] = fit_params[fit_param_names.index('alpha_thick')].median
            master_data.loc[dummy_idx, 'alpha_thick_m'] = fit_params[fit_param_names.index('alpha_thick')].minus
            master_data.loc[dummy_idx, 'alpha_thick_p'] = fit_params[fit_param_names.index('alpha_thick')].plus
            master_data.loc[dummy_idx, 'alpha_thin'] = fit_params[fit_param_names.index('alpha_thin')].median
            master_data.loc[dummy_idx, 'alpha_thin_m'] = fit_params[fit_param_names.index('alpha_thin')].minus
            master_data.loc[dummy_idx, 'alpha_thin_p'] = fit_params[fit_param_names.index('alpha_thin')].plus
            #get highest and lowest flux density
            master_data.loc[dummy_idx, 'max_obs_freq'] = flux_data['Frequency (Hz)'].max()
            master_data.loc[dummy_idx, 'min_obs_freq'] = flux_data['Frequency (Hz)'].min()

            #get number of points above and below the peak
            master_data.loc[dummy_idx, 'n_above_peak'] = flux_data[flux_data['Frequency (Hz)'] > fit_params[fit_param_names.index('peak_freq')].median*1e6].shape[0]
            master_data.loc[dummy_idx, 'n_below_peak'] = flux_data[flux_data['Frequency (Hz)'] < fit_params[fit_param_names.index('peak_freq')].median*1e6].shape[0]


            if 'retrig' in result_array[0].model_type:
                master_data.loc[dummy_idx, 'alpha_retrig'] = fit_params[fit_param_names.index('alpha_retrig')].median
                master_data.loc[dummy_idx, 'alpha_retrig_m'] = fit_params[fit_param_names.index('alpha_retrig')].minus
                master_data.loc[dummy_idx, 'alpha_retrig_p'] = fit_params[fit_param_names.index('alpha_retrig')].plus
                master_data.loc[dummy_idx, 'trough_flux'] = fit_params[fit_param_names.index('trough_flux')].median
                master_data.loc[dummy_idx, 'trough_flux_m'] = fit_params[fit_param_names.index('trough_flux')].minus
                master_data.loc[dummy_idx, 'trough_flux_p'] = fit_params[fit_param_names.index('trough_flux')].plus
                master_data.loc[dummy_idx, 'trough_freq'] = fit_params[fit_param_names.index('trough_freq')].median
                master_data.loc[dummy_idx, 'trough_freq_m'] = fit_params[fit_param_names.index('trough_freq')].minus
                master_data.loc[dummy_idx, 'trough_freq_p'] = fit_params[fit_param_names.index('trough_freq')].plus
                master_data.loc[dummy_idx, 'peak_outofrange'] = (fit_params[fit_param_names.index('peak_freq')].median > flux_data['Frequency (Hz)'].max()*1e6) or (fit_params[fit_param_names.index('peak_freq')].median < flux_data['Frequency (Hz)'].min()*1e6)
                master_data.loc[dummy_idx, 'trough_outofrange'] = (fit_params[fit_param_names.index('trough_freq')].median > flux_data['Frequency (Hz)'].max()*1e6) or (fit_params[fit_param_names.index('trough_freq')].median < flux_data['Frequency (Hz)'].min()*1e6)
        else:
            master_data.loc[dummy_idx, 'snorm'] = fit_params[fit_param_names.index('S_norm')].median
            master_data.loc[dummy_idx, 'snorm_m'] = fit_params[fit_param_names.index('S_norm')].minus
            master_data.loc[dummy_idx, 'snorm_p'] = fit_params[fit_param_names.index('S_norm')].plus

            master_data.loc[dummy_idx, 'alpha_thin'] = fit_params[fit_param_names.index('alpha')].median
            master_data.loc[dummy_idx, 'alpha_thin'] = fit_params[fit_param_names.index('alpha')].minus
            master_data.loc[dummy_idx, 'alpha_thin'] = fit_params[fit_param_names.index('alpha')].plus

        #now save!
        write_mode = ('a' if not args.overwrite else 'w')
        master_data.to_csv(args.write_output, mode = write_mode, header = False, index = False)


#otherwise we are reading master_data from a file
elif args.file:
    input_data = pd.read_csv(args.file, header = 0)
    print(input_data)
    if not (('RA' not in input_data.columns and 'Dec' not in input_data.columns) or \
         'IAU_designation' not in input_data.columns or 'NED_name' not in input_data.columns):
         print('''Format of input file not recognised. Please make sure it is in a format that can 
         be parsed by pandas, with at least one of:
         - the source RA and Dec (both in decimal degrees)
         - IAU_designation, or
         - NED_name''')
         exit()
    #loop through objects
    if 'IAU_designation' in input_data.columns:
        print('Using \'IAU_designation\' column...')
    elif 'NED_name' in input_data.columns:
        print('Using \'NED_Name\' column...')
    elif 'RA' in input_data.columns:
        print('Using \'RA\', \'Dec\' columns...')

    for src_idx in input_data.index.tolist():
        #get flux master_data
        if 'IAU_designation' in input_data.columns:
            src_iau_name, ra, dec, separation, racs_id = info.resolve_name_generic(iau_name = input_data.loc[src_idx, 'IAU_designation'])

        elif 'NED_name' in input_data.columns:
            racs_name, ra, dec = info.find_racs_src(ned_name = input_data.loc[src_idx, 'NED_name'])
            src_iau_name, src_ra, src_dec, separation, racs_id = info.resolve_name_generic(iau_name = racs_name)

        elif 'RA' in input_data.columns:
            racs_name, ra, dec = info.find_racs_src(ra = input_data.loc[src_idx, 'RA'], dec = input_data.loc[src_idx, 'Dec'])
            src_iau_name, src_ra, src_dec, separation, racs_id = info.resolve_name_generic(iau_name = racs_name)


        #get the flux master_data
        if args.use_local:
            flux_data, peak_flux_data, alma_variable, racs_id = parser.retrieve_fluxdata_local(racs_id = racs_id)
        elif not args.use_local and args.custom_data_file is None:
            flux_data, peak_flux_data, alma_variable = parser.retrieve_fluxdata_remote(iau_name = src_iau_name,
            racs_id = racs_id, ra=ra, dec=dec)
        else:
            flux_data = pd.read_csv(args.custom_data_file)

        #run fitting

        #other useful diagnostics
        # get auxiliary info about the source compactness and possible blending
        racs_n_gaus, racs_fluxratio = info.check_racs_compactness(src_name = src_iau_name) 
        gleam_blending_flag = info.check_confusion(src_name = src_iau_name)
        gleam_fluxratio, gleam_sep = info.check_gleam_compactness(src_name = src_iau_name)
        racs_n_gaus, racs_fluxratio = info.check_racs_compactness(src_name = src_iau_name) 
        at20g_compactness, at20g_visibility, at20g_sep = info.check_at20g_compactness(src_name = src_iau_name)

        #remove bottom GLEAM bands if there is likely confusion
        if gleam_blending_flag[0] == True:
            flux_data = flux_data[flux_data['Frequency (Hz)'] > 1e8]


        # now initialise fitter
        fitter.update_data(data=flux_data, peak_data=peak_flux_data, name=src_iau_name)

        # setup models depending on whether or not we require a GP (required if we have GLEAM master_data)
        if "GLEAM" in flux_data["Survey quickname"].tolist():
            with open("data/models/gp_modelset.pkl", "rb") as f:
                model_list = pickle.load(f)
        else:
            with open("data/models/modelset.pkl", "rb") as f:
                model_list = pickle.load(f)

        # tell me what models we are running!
        print("running RaiSED for {} using models:".format(src_iau_name))
        for model in model_list:
            print(model["model_type"])

        # do the fitting
        result_array = fitter.run_all_models(model_list)
        result_array, fit_params, log10z_arr = fitter.analyse_fits(result_array)
        model_type_arr = [result_array[x].model_type for x in range(len(result_array))]
        fit_param_names = [fit_params[x].name for x in range(len(fit_params))]

        #print out the parameters given by fitting two different ways
        print('Best fit model type: ', result_array[0].model_type)

        #plot if we are plotting
        if args.plot:
            # now get some plots!
            plotter.update_data(
                data=flux_data, peak_data=peak_flux_data, name=src_iau_name, savestr_end=""
            )
            plotter.update_results(result_array)
            plotter.plot_all_models()
            plotter.plot_epoch()
            plotter.plot_survey()
            plotter.plot_publication()
            plotter.plot_best_model()

    #write to file if we are writing
    #if we are writing to output, collect in a master_dataFrame and save
    if args.write_output:
        master_data.loc[dummy_idx, 'obj_name'] = racs_id
        master_data.loc[dummy_idx, 'iau_name'] = src_iau_name
        master_data.loc[dummy_idx, 'ra'] = src_ra
        master_data.loc[dummy_idx, 'dec'] = src_dec
        master_data.loc[dummy_idx, 'n_flux'] = flux_data.shape[0]
        master_data.loc[dummy_idx, 'peaked_spectrum'] = 'lin' not in result_array[0].model_type
        master_data.loc[dummy_idx, 'retrig_spectrum'] = 'retrig' in result_array[0].model_type
        master_data.loc[dummy_idx, 'racs_extended'] = racs_fluxratio
        master_data.loc[dummy_idx, 'n_gaus_racs'] =  racs_n_gaus
        master_data.loc[dummy_idx, 'n_flux'] = flux_data.shape[0]
        master_data.loc[dummy_idx, 'n_flux_new'] = ''
        master_data.loc[dummy_idx, 'new_only'] = False


        master_data.loc[dummy_idx, 'gleam_blended'] =  gleam_blending_flag
        master_data.loc[dummy_idx, 'gleam_fluxratio'] = gleam_fluxratio
        master_data.loc[dummy_idx, 'at20g_compact'] =  at20g_compactness
        master_data.loc[dummy_idx, 'at20g_visibility'] =  at20g_visibility

        #add racs flux
        master_data.loc[dummy_idx, 'racs_low_flux'] = flux_data.loc[flux_data['Survey quickname'] == 'RACS', 'Flux Density (Jy)'].values[0] 
        master_data.loc[dummy_idx, 'racs_low_flux_err'] = flux_data.loc[flux_data['Survey quickname'] == 'RACS', 'Uncertainty'].values[0]

        master_data.loc[dummy_idx, 'best_model'] = result_array[0].model_type
        master_data.loc[dummy_idx, 'best_model_log10Z'] = log10z_arr[0]
        if "GLEAM" in flux_data["Survey quickname"].tolist():
            master_data.loc[dummy_idx, 'linear_gp_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'lin' in s][0]]
            master_data.loc[dummy_idx, 'orienti_gp_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'orient' in s][0]]
            master_data.loc[dummy_idx, 'snellen_gp_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'snellen' in s][0]]
            master_data.loc[dummy_idx, 'retrig_gp_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'retrig' in s][0]]

            master_data.loc[dummy_idx, 'linear_nested_log10z'] = np.nan
            master_data.loc[dummy_idx, 'orienti_nested_log10z'] = np.nan
            master_data.loc[dummy_idx, 'snellen_nested_log10z'] = np.nan
            master_data.loc[dummy_idx, 'retrig_nested_log10z'] = np.nan
        
        else:
            master_data.loc[dummy_idx, 'linear_nested_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'lin' in s][0]]
            master_data.loc[dummy_idx, 'orienti_nested_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'orient' in s][0]]
            master_data.loc[dummy_idx, 'snellen_nested_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'snell' in s][0]]
            master_data.loc[dummy_idx, 'retrig_nested_log10z'] = log10z_arr[[idx for idx, s in enumerate(model_type_arr) if 'retrig' in s][0]]
            
            master_data.loc[dummy_idx, 'linear_gp_log10z'] = np.nan
            master_data.loc[dummy_idx, 'orienti_gp_log10z'] = np.nan
            master_data.loc[dummy_idx, 'snellen_gp_log10z'] = np.nan
            master_data.loc[dummy_idx, 'retrig_gp_log10z'] = np.nan


        if 'lin' not in result_array[0].model_type:
            master_data.loc[dummy_idx, 'peak_flux'] = fit_params[fit_param_names.index('peak_flux')].median
            master_data.loc[dummy_idx, 'peak_flux_m'] = fit_params[fit_param_names.index('peak_flux')].minus
            master_data.loc[dummy_idx, 'peak_flux_p'] = fit_params[fit_param_names.index('peak_flux')].plus
            master_data.loc[dummy_idx, 'peak_freq'] = fit_params[fit_param_names.index('peak_freq')].median
            master_data.loc[dummy_idx, 'peak_freq_m'] = fit_params[fit_param_names.index('peak_freq')].minus
            master_data.loc[dummy_idx, 'peak_freq_p'] = fit_params[fit_param_names.index('peak_freq')].plus
            master_data.loc[dummy_idx, 'alpha_thick'] = fit_params[fit_param_names.index('alpha_thick')].median
            master_data.loc[dummy_idx, 'alpha_thick_m'] = fit_params[fit_param_names.index('alpha_thick')].minus
            master_data.loc[dummy_idx, 'alpha_thick_p'] = fit_params[fit_param_names.index('alpha_thick')].plus
            master_data.loc[dummy_idx, 'alpha_thin'] = fit_params[fit_param_names.index('alpha_thin')].median
            master_data.loc[dummy_idx, 'alpha_thin_m'] = fit_params[fit_param_names.index('alpha_thin')].minus
            master_data.loc[dummy_idx, 'alpha_thin_p'] = fit_params[fit_param_names.index('alpha_thin')].plus
            #get highest and lowest flux density
            master_data.loc[dummy_idx, 'max_obs_freq'] = flux_data['Frequency (Hz)'].max()
            master_data.loc[dummy_idx, 'min_obs_freq'] = flux_data['Frequency (Hz)'].min()

            #get number of points above and below the peak
            master_data.loc[dummy_idx, 'n_above_peak'] = flux_data[flux_data['Frequency (Hz)'] > fit_params[fit_param_names.index('peak_freq')].median*1e6].shape[0]
            master_data.loc[dummy_idx, 'n_below_peak'] = flux_data[flux_data['Frequency (Hz)'] < fit_params[fit_param_names.index('peak_freq')].median*1e6].shape[0]


            if 'retrig' in result_array[0].model_type:
                master_data.loc[dummy_idx, 'alpha_retrig'] = fit_params[fit_param_names.index('alpha_retrig')].median
                master_data.loc[dummy_idx, 'alpha_retrig_m'] = fit_params[fit_param_names.index('alpha_retrig')].minus
                master_data.loc[dummy_idx, 'alpha_retrig_p'] = fit_params[fit_param_names.index('alpha_retrig')].plus
                master_data.loc[dummy_idx, 'trough_flux'] = fit_params[fit_param_names.index('trough_flux')].median
                master_data.loc[dummy_idx, 'trough_flux_m'] = fit_params[fit_param_names.index('trough_flux')].minus
                master_data.loc[dummy_idx, 'trough_flux_p'] = fit_params[fit_param_names.index('trough_flux')].plus
                master_data.loc[dummy_idx, 'trough_freq'] = fit_params[fit_param_names.index('trough_freq')].median
                master_data.loc[dummy_idx, 'trough_freq_m'] = fit_params[fit_param_names.index('trough_freq')].minus
                master_data.loc[dummy_idx, 'trough_freq_p'] = fit_params[fit_param_names.index('trough_freq')].plus
                master_data.loc[dummy_idx, 'peak_outofrange'] = (fit_params[fit_param_names.index('peak_freq')].median > flux_data['Frequency (Hz)'].max()*1e6) or (fit_params[fit_param_names.index('peak_freq')].median < flux_data['Frequency (Hz)'].min()*1e6)
                master_data.loc[dummy_idx, 'trough_outofrange'] = (fit_params[fit_param_names.index('trough_freq')].median > flux_data['Frequency (Hz)'].max()*1e6) or (fit_params[fit_param_names.index('trough_freq')].median < flux_data['Frequency (Hz)'].min()*1e6)
        else:
            master_data.loc[dummy_idx, 'snorm'] = fit_params[fit_param_names.index('S_norm')].median
            master_data.loc[dummy_idx, 'snorm_m'] = fit_params[fit_param_names.index('S_norm')].minus
            master_data.loc[dummy_idx, 'snorm_p'] = fit_params[fit_param_names.index('S_norm')].plus

            master_data.loc[dummy_idx, 'alpha_thin'] = fit_params[fit_param_names.index('alpha')].median
            master_data.loc[dummy_idx, 'alpha_thin'] = fit_params[fit_param_names.index('alpha')].minus
            master_data.loc[dummy_idx, 'alpha_thin'] = fit_params[fit_param_names.index('alpha')].plus

        #increment dummy index for writing output
        dummy_idx += 1
    
    #now save!
    write_mode = ('a' if not args.overwrite else 'w')
    master_data.to_csv(args.write_output, mode = write_mode, header = False, index = False)

