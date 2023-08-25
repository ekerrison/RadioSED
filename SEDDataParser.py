import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm
import astropy.units as u
import astropy.coordinates as coord
from astroquery.vizier import Vizier
'''A class that parses SED data. It will either load in the masterlists (from a specified 
directory) and pull pandas dataframes out of these, or it will create the dataframse by
scraping Vizier for data for an individual source'''

#silence warning about pandas copies
pd.options.mode.chained_assignment = None

class SEDDataParser:

    def __init__(self, use_local = True, masterlist_filepath = './data/allsky_masterlist.csv', \
        peak_masterlist_filepath = './data/allsky_peak_masterlist.csv', \
        almacal_masterlist_filepath = './data/allsky_almacal_masterlist.csv', survey_info_filepath = './data/included_surveys_final.csv'):
        
        self.survey_info = pd.read_csv(survey_info_filepath, header = 0)

        #check survey info does not have any NaNs, these are bad!

        if use_local:
            ################################################################################
            #read in the master file
            self.masterlist = pd.read_csv(masterlist_filepath, header = 0)
            self.peak_masterlist = pd.read_csv(peak_masterlist_filepath, header = 0)

            #plus the almacal masterlist!
            self.almacal_masterlist = pd.read_csv(almacal_masterlist_filepath, header = 0)

            #####################################

            #init tqdm for nice progress bars
            tqdm.pandas(desc='parsing frequencies')

            #print(masterlist.columns)
            print('initialising masterlist data, please be patient...')
            #convert survey_fluxes, survey_freqs, survey_bibcodes, survey_flux_errs
            self.masterlist['survey_fluxes'] = self.masterlist['survey_fluxes'].progress_apply(lambda x: literal_eval(x.replace('nan', 'None')))
            
            tqdm.pandas(desc='parsing fluxes')
            self.masterlist['survey_freqs'] = self.masterlist['survey_freqs'].progress_apply(lambda x: literal_eval(x))
            
            tqdm.pandas(desc='parsing flux errors')
            self.masterlist['survey_flux_errs'] = self.masterlist['survey_flux_errs'].progress_apply(lambda x: literal_eval(x.replace('nan', 'None')))
            
            tqdm.pandas(desc='parsing bibcodes')
            self.masterlist['survey_bibcodes'] = self.masterlist['survey_bibcodes'].progress_apply(lambda x: literal_eval(x))
            
            tqdm.pandas(desc='parsing survey names')
            self.masterlist['survey_names'] = self.masterlist['survey_names'].progress_apply(lambda x: literal_eval(x))
            
            tqdm.pandas(desc='parsing vizier locations')
            self.masterlist['survey_vizier'] = self.masterlist['survey_vizier'].progress_apply(lambda x: literal_eval(x))

            #convert survey_fluxes, survey_freqs, survey_bibcodes, survey_flux_errs
            tqdm.pandas(desc='parsing peak fluxes')
            self.peak_masterlist['survey_fluxes'] = self.peak_masterlist['survey_fluxes'].progress_apply(lambda x: literal_eval(x.replace('nan', 'None')))
            
            tqdm.pandas(desc='parsing peak frequencies')
            self.peak_masterlist['survey_freqs'] = self.peak_masterlist['survey_freqs'].progress_apply(lambda x: literal_eval(x))
            
            tqdm.pandas(desc='parsing peak flux errors')
            self.peak_masterlist['survey_flux_errs'] = self.peak_masterlist['survey_flux_errs'].progress_apply(lambda x: literal_eval(x.replace('nan', 'None')))
            
            tqdm.pandas(desc='parsing peak bibcodes')
            self.peak_masterlist['survey_bibcodes'] = self.peak_masterlist['survey_bibcodes'].progress_apply(lambda x: literal_eval(x))
            
            tqdm.pandas(desc='parsing peak names')
            self.peak_masterlist['survey_names'] = self.peak_masterlist['survey_names'].progress_apply(lambda x: literal_eval(x))
            
            tqdm.pandas(desc='parsing peak vizier locations')
            self.peak_masterlist['survey_vizier'] = self.peak_masterlist['survey_vizier'].progress_apply(lambda x: literal_eval(x))
            print('=== masterlists ready ===')
        else:
            print('SEDDataParser initialised for scraping data from the web only. If you want to use local data please re-initialise the instance of this class, passing in \'use_local = True\'')
        
        return

    def retrieve_fluxdata_local(self, iau_name = None, racs_id = None):
        #extract from masterlist
        datarow = self.masterlist.loc[self.masterlist['source_id'] == racs_id, :]

        #same for peak data!
        peak_datarow = self.peak_masterlist.loc[self.peak_masterlist['source_id'] == racs_id, :]

        #if more than one, just pick first one THIS WILL ONLY BE AN ISSUE IF IT IS A COMPLEX SOURCE!
        # and in that case it will be flagged anyway!
        if peak_datarow.shape[0] > 1:
            print(peak_datarow[['survey_fluxes', 'survey_names']])
            raise ValueError('there should only be 1 matching peak flux file!!')
            peak_datarow = peak_datarow.loc[peak_datarow.index.tolist()[0],:]
        else:
            peak_datarow = peak_datarow.squeeze()


        #if uncertainties not the same length as fluxes, check for CCA survey, which doesn't have errors on Vizier  
        if len(datarow['survey_fluxes']) != len(datarow['survey_flux_errs']):
            #raise ValueError('mismatch length beween fluxes and uncertainties!')
            if 'CCA' in datarow['survey_names']:
                #if CCA then insert a 10% flux error
                first_surv_idx = datarow['survey_names'].index('CCA')
                datarow['survey_flux_errs'].insert(first_surv_idx, datarow['survey_fluxes'][first_surv_idx]*0.1)
                # and second CCA flux 
                datarow['survey_flux_errs'].insert(first_surv_idx+1, datarow['survey_fluxes'][first_surv_idx+1]*0.1)
            else:
                print(datarow['survey_fluxes'], datarow['survey_flux_errs'])
                print(datarow['survey_names'])
                raise Exception('flux lengths do not match when building table!')

        #now make into a flux table!
        flux_data = pd.DataFrame({'Frequency (Hz)': datarow['survey_freqs'].squeeze(), \
        'Flux Density (Jy)': datarow['survey_fluxes'].squeeze(), 'Uncertainty' : datarow['survey_flux_errs'].squeeze(), \
        'Survey quickname' : datarow['survey_vizier'].squeeze(), 'Refcode' : datarow['survey_bibcodes'].squeeze(), \
        'Survey quickname': datarow['survey_names'].squeeze()})
        

        #drop vlssr as it is peak flux!
        if 'vlssr' in flux_data['Survey quickname'].apply(lambda x: x.lower()).tolist():
            #get index and drop because it's peak!
            drop_idx = flux_data['Survey quickname'].apply(lambda x: x.lower()).tolist().index('vlssr')
            flux_data = flux_data.drop(index = drop_idx)
            flux_data = flux_data.reset_index(drop = True)

        #add racs info
        append_flux_idx = flux_data.shape[0]
        flux_data.loc[append_flux_idx, 'Frequency (Hz)'] = 888000000
        flux_data.loc[append_flux_idx, 'Flux Density (Jy)'] = datarow['total_flux_source'].values[0]/1e3
        flux_data.loc[append_flux_idx, 'Uncertainty'] = datarow['e_total_flux_source'].values[0]/1e3
        flux_data.loc[append_flux_idx, 'Survey quickname'] = ''
        flux_data.loc[append_flux_idx, 'Refcode'] = '2021PASA...38...58H'
        flux_data.loc[append_flux_idx, 'Survey quickname'] = 'RACS'

        #now add ALMA calibrator data if available!
        almacal_pts = self.almacal_masterlist[self.almacal_masterlist['source_id'] == racs_id]

        #if alma has multiple fluxes at the same frequency (to the nearest 5ghz) that vary by more than 10 sigma, flag
        # this source as variable, and don't include fluxes in fitting
        alma_variable = np.nan
        if almacal_pts.shape[0] > 0:
            almacal_pts['Freq_rounded'] = almacal_pts['Freq'].apply(lambda x: 5 * round(x/5))
            max_flux_diff = almacal_pts.groupby('Freq_rounded')['Flux'].agg(np.ptp)
            max_err = almacal_pts.groupby('Freq_rounded')['e_Flux'].max()
            if (max_flux_diff > 10*max_err).any():
                alma_variable = True
            else:
                alma_variable = False
                #add almacal to flux_data
                alma_count = 0
                for alma_idx in almacal_pts.index.tolist():
                    flux_data.loc[append_flux_idx+1+alma_count, 'Frequency (Hz)'] = almacal_pts.loc[alma_idx,'Freq']*1e9
                    flux_data.loc[append_flux_idx+1+alma_count, 'Flux Density (Jy)'] = almacal_pts.loc[alma_idx,'Flux']
                    flux_data.loc[append_flux_idx+1+alma_count,'Survey quickname'] = 'ALMACAL'
                    flux_data.loc[append_flux_idx+1+alma_count,'Uncertainty'] = almacal_pts.loc[alma_idx,'e_Flux']
                    flux_data.loc[append_flux_idx+1+alma_count,'Survey quickname'] = 'J/MNRAS/485/1188/acccat'
                    flux_data.loc[append_flux_idx+1+alma_count,'Refcode'] =  '2019MNRAS.485.1188B'
                    alma_count += 1

        #resort by frequency
        flux_data = flux_data.sort_values(by = 'Frequency (Hz)')

        #remove nan frequencies
        flux_data = flux_data[~pd.isnull(flux_data['Flux Density (Jy)'])]

        #remove negative fluxes - shouldn't be necessary but sometimes GLEAM forced
        #fluxes drop to negatives
        flux_data = flux_data[flux_data['Flux Density (Jy)'] > 0]

        if len(peak_datarow['survey_freqs']) > 0:
            #now make into a flux table!
            peak_flux_data = pd.DataFrame({'Frequency (Hz)': peak_datarow['survey_freqs'], \
            'Flux Density (Jy)': peak_datarow['survey_fluxes'], 'Uncertainty' : peak_datarow['survey_flux_errs'], \
            'Survey quickname' : peak_datarow['survey_vizier'], 'Refcode' : peak_datarow['survey_bibcodes'], \
            'Survey quickname': peak_datarow['survey_names']})
        else:
            peak_flux_data = pd.DataFrame(columns = ['Frequency (Hz)', 'Flux Density (Jy)', 'Uncertainty', \
                'Survey quickname', 'Refcode', 'Survey quickname'])

        if peak_flux_data.shape[0] > 1:
            #add racs info
            append_peak_idx = peak_flux_data.shape[0] + 1
            peak_flux_data.loc[append_peak_idx, 'Frequency (Hz)'] = 888000000
            peak_flux_data.loc[append_peak_idx, 'Flux Density (Jy)'] = peak_datarow['peak_flux']/1e3
            peak_flux_data.loc[append_peak_idx, 'Uncertainty'] = peak_datarow['e_peak_flux']/1e3
            peak_flux_data.loc[append_peak_idx, 'Survey quickname'] = ''
            peak_flux_data.loc[append_peak_idx, 'Refcode'] = '2021PASA...38...58H'
            peak_flux_data.loc[append_peak_idx, 'Survey quickname'] = 'RACS'

            #resort by frequency
            peak_flux_data = peak_flux_data.sort_values(by = 'Frequency (Hz)')
        else:
            #add racs info
            peak_flux_data['Flux Density (Jy)'] = peak_datarow['peak_flux']/1e3
            peak_flux_data['Uncertainty'] = peak_datarow['e_peak_flux']/1e3
            peak_flux_data['Survey quickname'] = ''
            peak_flux_data['Refcode'] = '2021PASA...38...58H'
            peak_flux_data['Frequency (Hz)'] = 888*1e6
            peak_flux_data['Survey quickname'] = 'RACS'
        return flux_data, peak_flux_data, alma_variable


    def query_vizier(self, cat, ra, dec, columns, radius):
    #radius to search (arcsec)
        rad = radius/3600
        coords = coord.SkyCoord(ra=ra, dec = dec,  unit=(u.deg, u.deg),frame='icrs')
        if columns == None:
            result = Vizier.query_region(coords, radius = rad*u.deg, catalog = cat)
        else:
            viz_instance = Vizier(columns = ["**"])#columns, catalog = cat)# <-- what we currently have just gets all cols!
            result = viz_instance.query_region(coords, radius = rad*u.deg, catalog = cat)
        return result

    def retrieve_fluxdata_remote(self, iau_name = None, racs_id = None, ra = None, dec = None, reliable_only = True, use_island_radius = True):
        '''
        Essentially a wrapper for astroquery's Vizier.query_region() function.
        Output is a pandas table. Takes ra,dec coordinates as input
        '''

        photometry_table = pd.DataFrame(columns = ['Frequency (Hz)', 'Flux Density (Jy)', 'Uncertainty', 'Survey quickname', 'Refcode'])

        peak_phot_table = pd.DataFrame(columns = ['Frequency (Hz)', 'Flux Density (Jy)', 'Uncertainty', 'Survey quickname', 'Refcode'])

        #filter out so we only use survey we want
        if reliable_only:
            rel_threshold = 1
        else:
            rel_threshold = 99
        surveys_used = self.survey_info[self.survey_info['reliability'] <= rel_threshold]

        count = 1
        #first,query object
        for idx in surveys_used.index.tolist():
            
            radius = surveys_used.loc[idx, 'match_radius_99']

            #SKIP IF ALMACAL
            if surveys_used.loc[idx, 'Name'] == 'ALMACAL':
                alma_cat = surveys_used.loc[idx, 'Vizier name']
                alma_columns = ['Flux', 'e_Flux', 'Band']
                alma_radius = radius
                alma_refcode = surveys_used.loc[idx, 'bibcode']
                continue


            #skip if RACS, we already have this presumably?
            if surveys_used.loc[idx, 'Name'] == 'RACS':
                res = Vizier.query_constraints(catalog = surveys_used.loc[idx, 'Vizier name'], ID= '={}'.format(racs_id))[0].to_pandas()
                
                single_row = pd.DataFrame({'Frequency (Hz)': [float(surveys_used.loc[idx, 'frequencies (MHz)'])], 'Flux Density (Jy)': [res['Ftot'].values[0]/1e3], 'Uncertainty': [res['e_Ftot'].values[0]/1e3], 'Survey quickname': [surveys_used.loc[idx,'Vizier name']], 'Refcode':[surveys_used.loc[idx,'bibcode']], 'Survey quickname': [surveys_used.loc[idx,'Name']]})
                photometry_table = pd.concat([photometry_table, single_row], ignore_index = True, axis = 0)
                
                peak_single_row = pd.DataFrame({'Frequency (Hz)': [float(surveys_used.loc[idx, 'frequencies (MHz)'])], 'Flux Density (Jy)': [res['Fpk'].values[0]/1e3], 'Uncertainty': [res['e_Fpk'].values[0]/1e3], 'Survey quickname': [surveys_used.loc[idx,'Vizier name']], 'Refcode':[surveys_used.loc[idx,'bibcode']], 'Survey quickname': [surveys_used.loc[idx,'Name']]})
                peak_phot_table = pd.concat([peak_phot_table, peak_single_row], ignore_index = True, axis = 0)

                continue

            #get column names to return from Vizier query
            if ';' in surveys_used.loc[idx, 'flux_columns'] and not pd.isnull(surveys_used.loc[idx, 'e_flux_columns']):
                colnames = surveys_used.loc[idx, 'flux_columns'].split(';') + surveys_used.loc[idx, 'e_flux_columns'].split(';')
            elif not pd.isnull(surveys_used.loc[idx, 'e_flux_columns']):
                colnames = surveys_used.loc[idx, 'flux_columns'] + surveys_used.loc[idx, 'e_flux_columns']
            else:
                colnames = surveys_used.loc[idx, 'flux_columns'].split(';')
            cat_table = self.query_vizier(surveys_used.loc[idx, 'Vizier name'], ra, dec, columns = colnames, radius = radius)
         
            if len(cat_table) == 0:
                count += 1
                continue
            #because query_vizier returns a list of catalogues, but we are only querying 1 at a time
            cat_table = cat_table[0]

        #now append photometry val if it exists
            if len(cat_table) == 0:
                #here the table was empty ... not sure I need this, I think it's covered
                #above? leave it in anyway, I'm too lazy to check...
                count += 1
                continue
            else:
                #now we add the new photometry values, we want to fill 'NED Uncertainty',
                #Frequency, and Flux Density
                if len(cat_table) > 1:
                    #if VLASS, check whether multiple entries are duplicates, and if they are pick the preferred one
                    if (surveys_used.loc[idx,'Name'] == 'VLASS') and ( len(np.unique(cat_table['CompName'])) == 1):
                        cat_table = cat_table[cat_table['DupFlag'] == 1]
                    else:
                        print('MORE THAN 1 ENTRY FOUND WITHIN 99% MATCH RADIUS FOR THIS OBJECT')
                        cat_table.pprint(max_lines=-1, max_width=-1)
                        cat_table=cat_table[cat_table['_r'] == np.min(cat_table['_r'])]
                        #exit()

                #otherwise extract the useful information from the prefilled survey table
                cat_freqs = surveys_used.loc[idx, 'frequencies (MHz)'].split(';')
                cat_fluxes = surveys_used.loc[idx, 'flux_columns'].split(';')
                cat_flux_errs = surveys_used.loc[idx, 'e_flux_columns'].split(';')

                #loop through all the frequencies in this survey, and extract data from the query
                for i in range(len(cat_fluxes)):

                    #assign frequency - this allows us to have peak and integrated at the same frequency!
                    if len(cat_freqs) == 1:
                        current_freq = cat_freqs[0]
                    else:
                        current_freq = cat_freqs[i]

                    current_flux_col = cat_fluxes[i]
                    current_flux_val = cat_table[current_flux_col].data[0]

                    #if C1813_new then skip unless marked as new measurement!
                    if surveys_used.loc[idx,'Name'] == 'C1813_new' and cat_table['n_' + current_flux_col] != '*':
                        print(cat_table['n_' + current_flux_col])
                        print('no')
                        continue

                    #if actual error column exists, get it
                    if cat_flux_errs != '':
                        current_err_col = cat_flux_errs[i]
                        current_flux_err = cat_table[current_err_col].data[0]
                    else:
                        current_flux_err = cat[9]*cat_table[current_flux_col].data[0]

                    #if vlass, bump up err! - no longer needed as of version 2, but this is not currently on Vizier!
                    if surveys_used.loc[idx,'Name'] == 'VLASS':
                        current_err_col = cat_flux_errs[i]
                        current_flux_err = cat[9]*cat_table[current_flux_col].data[0] + cat_table[current_err_col].data[0]

                    #account for cats where measurement is in mJy
                    if surveys_used.loc[idx,'flux units'] == 'mJy':
                        current_flux_val = current_flux_val/1000
                        current_flux_err = current_flux_err/1000
                    
                    #make sure NOT VLSSr which is peak only:
                    if not surveys_used.loc[idx,'Name'] == 'VLSSr':
                    #append to table to return
                        single_row = pd.DataFrame({'Frequency (Hz)': [current_freq], 'Flux Density (Jy)': [current_flux_val], 'Uncertainty': [current_flux_err], 'Survey quickname': [surveys_used.loc[idx,'Vizier name']], 'Refcode':[surveys_used.loc[idx,'bibcode']], 'Survey quickname': [surveys_used.loc[idx,'Name']]})
                        photometry_table = pd.concat([photometry_table, single_row], ignore_index = True, axis = 0)

    ##############################################################################################

                    #now do the same for the peak flux if it exists
                    #get column names for peak fluxes
                    if not pd.isnull(surveys_used.loc[idx, 'peak_fluxcol']):
                        peak_colnames = surveys_used.loc[idx, 'peak_fluxcol'].split(';') + surveys_used.loc[idx, 'peak_efluxcol'].split(';')
                        peak_cat_table = self.query_vizier(surveys_used.loc[idx, 'Vizier name'], ra, dec, columns = peak_colnames, radius = radius)
                        if len(cat_table) == 0:
                            continue
                        #because query_vizier returns a list of catalogues, but we are only querying 1 at a time
                        peak_cat_table = peak_cat_table[0]

                    #now append photometry val if it exists
                        if len(peak_cat_table) == 0:
                            continue

                        #if more than one match within the radius only append the closest
                        if len(peak_cat_table) > 1:
                            #if VLASS, check whether multiple entries are duplicates, and if they are pick the preferred one
                            if (surveys_used.loc[idx,'Name'] == 'VLASS') and ( len(np.unique(peak_cat_table['CompName'])) == 1):
                                peak_cat_table = peak_cat_table[peak_cat_table['DupFlag'] == 1]
                                #print(cat_table)
                                #exit()
                            else:
                                #print('MORE THAN 1 ENTRY FOUND WITHIN 99% MATCH RADIUS FOR THIS OBJECT')
                                #peak_cat_table.pprint(max_lines=-1, max_width=-1)
                                peak_cat_table=peak_cat_table[peak_cat_table['_r'] == np.min(peak_cat_table['_r'])]
                                #exit()

                        #otherwise extract the useful information from the prefilled survey table
                        peak_cat_freqs = surveys_used.loc[idx, 'frequencies (MHz)'].split(';')
                        peak_cat_fluxes = surveys_used.loc[idx, 'peak_fluxcol'].split(';')
                        peak_cat_flux_errs = surveys_used.loc[idx, 'peak_efluxcol'].split(';')

                        #read and append
                        #loop through all the frequencies in this survey, and extract data from the query
                        for i in range(len(peak_cat_fluxes)):

                            #assign frequency - this allows us to have peak and integrated at the same frequency!
                            if len(peak_cat_freqs) == 1:
                                peak_current_freq = peak_cat_freqs[0]
                            else:
                                peak_current_freq = peak_cat_freqs[i]

                            peak_current_flux_col = peak_cat_fluxes[i]
                            peak_current_flux_val = peak_cat_table[peak_current_flux_col].data[0]

                            #print('separation for {}'.format(surveys_used.loc[idx,'Name']))
                            #print(cat_table['_r'].value[0])

                            #if C1813_new then skip unless marked as new measurement!
                            if surveys_used.loc[idx,'Name'] == 'C1813_new' and peak_at_table['n_' + peak_current_flux_col] != '*':
                                print(peak_cat_table['n_' + peak_current_flux_col])
                                print('no')
                                continue

                            #if actual error column
                            if peak_cat_flux_errs != '':
                                peak_current_err_col = peak_cat_flux_errs[i]
                                #print(cat[0])
                                #print(peak_cat_table)
                                peak_current_flux_err = peak_cat_table[peak_current_err_col].data[0]
                            else:
                                peak_current_flux_err = cat[9]*peak_cat_table[peak_current_flux_col].data[0]

                            #if vlass, bump up err!
                            if surveys_used.loc[idx,'Name'] == 'VLASS':
                                peak_current_flux_err = cat[9]*peak_cat_table[peak_current_flux_col].data[0] + peak_cat_table[peak_current_err_col].data[0]

                            #fix mJy fluxes to be Jy
                            if surveys_used.loc[idx,'flux units'] == 'mJy':
                                peak_current_flux_val = peak_current_flux_val/1000
                                peak_current_flux_err = peak_current_flux_err/1000

                            #print('before')
                            #print(peak_phot_table)
                            peak_single_row = pd.DataFrame({'Frequency (Hz)': [peak_current_freq], 'Flux Density (Jy)': [peak_current_flux_val], 'Uncertainty': [peak_current_flux_err], 'Survey quickname': [surveys_used.loc[idx,'Vizier name']], 'Refcode':[surveys_used.loc[idx,'bibcode']], 'Survey quickname': [surveys_used.loc[idx,'Name']]})
                            peak_phot_table = pd.concat([peak_phot_table, peak_single_row], ignore_index = True, axis = 0)
                            #print('after')
                            #(peak_phot_table)

        #now add almacal if it exists!
        alma_fluxes = self.query_vizier(cat = alma_cat, ra = ra, dec = dec, columns = alma_columns, radius = alma_radius)[0].to_pandas()
        flux_idx = photometry_table.shape[0]
        if alma_fluxes.shape[0] > 0:
            for alma_idx in alma_fluxes.index.tolist():
                photometry_table.loc[flux_idx + alma_idx + 1, 'Frequency (Hz)'] = alma_fluxes.loc[alma_idx, 'Freq']*1e3
                photometry_table.loc[flux_idx + alma_idx + 1, 'Flux Density (Jy)'] = alma_fluxes.loc[alma_idx, 'Flux']
                photometry_table.loc[flux_idx + alma_idx + 1, 'Uncertainty'] = alma_fluxes.loc[alma_idx, 'e_Flux']
                photometry_table.loc[flux_idx + alma_idx + 1, 'Survey quickname'] = 'ALMACAL'
                photometry_table.loc[flux_idx + alma_idx + 1, 'Refcode'] = alma_refcode
                

        #now sort the table so that it is in ascending frequency order
        photometry_table['Frequency (Hz)'] = photometry_table['Frequency (Hz)'].astype(float)*1e6
        peak_phot_table['Frequency (Hz)'] = peak_phot_table['Frequency (Hz)'].astype(float)*1e6
        photometry_table.sort_values(by = 'Frequency (Hz)', inplace = True)
        peak_phot_table.sort_values(by = 'Frequency (Hz)', inplace = True)
        
        return photometry_table, peak_phot_table


    def add_survey_radius(self, survey_viz):
        '''Function to add info automatically to the list of included surveys. Currently
        this only works for surveys hosted on Vizier, but could be easily extended to those
        on CIRADA. Anything else is only for the adventurous!'''

        #NOTE that you will need to add the other info manually - there's no nice way to scrape
        # all of this from the catalogue info online.

        return