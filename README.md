# RadioSED

[![DOI](https://zenodo.org/badge/682385260.svg)](https://zenodo.org/badge/latestdoi/682385260)

## A Bayesian Nested Sampling approach to radio SED fitting for AGN.

This package uses nested sampling ([Skilling 2004](https://doi.org/10.1063/1.1835238)) to perform a Bayesian analysis of radio SEDs constructed from radio flux density measurements
obtained as part of large area surveys (or in some limited cases, as part of targeted followup campaigns). It is a pure Python implementation, and is essentially a wrapper around [Bilby](https://lscsoft.docs.ligo.org/bilby/#),
 the Bayesian inference library developed for Gravitational-wave astronomy. It makes use of [Dynesty](https://dynesty.readthedocs.io/en/latest/index.html) to perform the sampling steps,
 though other samplers could also be used thanks to Bilby's architecture.

Users can make use of a pre-defined set of models and surveys from which to draw
flux density measurements, or they can define their own models, and provide their own input flux density measurements. All flux density measurements are referenced against the RACS-LOW survey 
([Hale et al. 2021](https://ui.adsabs.harvard.edu/abs/2021PASA...38...58H/abstract)), and source names and IDs from the survey catalogue are used as identifiers.

## Models

At present, 4 models are implemented as part of this package. These are: 
- A simple power law
- A log-space parabola to capture curvature (as described in e.g. [Dallacasa et al. 2000](https://ui.adsabs.harvard.edu/abs/2000A%26A...363..887D/abstract) and [Orienti et al. 2007](https://ui.adsabs.harvard.edu/abs/2007A%26A...461..923O/abstract))
- A functional form for a peaked spectrum source, which reduces to a synchrotron self-absorbed source under the condition `k=2.5` ([Snellen et al. 1998](10.1051/aas:1998281))
- A 'retriggered' model comprising the functional form from [Snellen et al. 1998](10.1051/aas:1998281) combined with an additional power law component

Additional, user-defined models can be created by making a custom functional form, and a custom prior dictionary to be fed into the `SEDPriors` class using the `custom_prior()` method.
This can then be fed into the `Fitter` class much like the pre-defined models.

## Structure

All of the code has designed to be fairly modular. You can use the builtin `retrieve_fluxdata_local()` and `retrieve_fluxdata_remote()` functions to build up a radio SED using a pre-defined set of radio
surveys, or you can feed in your own, custom dataset. If you choose to feed in your own, the code expects a Pandas DataFrame object, with columns labelled: 
- Frequency (Hz)
- Flux Density (Jy)
- Uncertainty
- Survey quickname
- Refcode

Where the 'survey quickname' is some easily identifiable name (e.g. RACS) and the Refcode is the bibcode for the relevant survey paper. The Refcode should begin with the year of publication as an integer (e.g. 2017MNRAS.464.1146H) to allow RadioSED to parse it for the survey epoch (used in plotting).

More documentation coming soon!

## Setup

First clone RadioSED into a working directory of your choosing:
> `git clone https://github.com/ekerrison/RadioSED.git`

RadioSED is tested with Python 3.10. If you are familiar with Anaconda environments I recommend setting up a new one:
> `conda create --name radiosed python=3.10`
> `conda activate radiosed`

Then you will need to install all of the dependencies for RadioSED. At the simplest level there are four main packages required:
- [Bilby](https://lscsoft.docs.ligo.org/bilby/#) (version 2.1 for gauranteed compatibility)
- [George](https://george.readthedocs.io/en/latest/)
- [Dynesty](https://dynesty.readthedocs.io/en/latest/index.html)
- [Astroquery](https://astroquery.readthedocs.io/en/latest/)

These can be downloaded using your favourite package manager. If you are using Anaconda/Conda this can be achieved in a single line:
> `conda install -c conda-forge george bilby=2.1 astroquery`
(note that installing bilby will also install the correct version of dynesty).

A comprehensive list of package requirements can be found in `requirements.txt` (for pip-compatible formatting) or `environment.yml` (for anaconda-style formatting).  
If you are running anaconda or miniconda, once you clone this repo you can run `conda env create -n radiosed -f environment.yml` from within the folder to create a new environment
with all the packages required to run RadioSED.

You will also need decompress some data files after you have cloned the repository. These are the `allsky_peak_masterlist.tar.gz` and `allsky_masterlist.tar.gz` files located within the `data\` subdirectory.
This can be achieved with the following commands from the main RadioSED directory:
> `tar -xvf data/allsky_masterlist.tar.gz -C data`
> `tar -xvf data/allsky_peak_masterlist.tar.gz -C data`

If you would like to make use of the pre-determined crossmatches that come with RadioSED, you will need to initialise the DataParser with option `use_local=True`.

## Usage

To see how RadioSED is run, please take a look at the `run_radiosed.py` script. This can be used as the main script from which you run fitting on your own sources, or it can
be modified to suit your needs. There is also the option to use RadioSED with a command line interface by calling `radiosed.py`. Call `python radiosed.py -h` for help and usage options.

Please note that to perform the default, 4-model inference on one source can take between 5-15 minutes depending on the specifications of the machine you are using.

## Examples
There are some example data and input files located in the `example_files/` directory. They can be used to understand several RadioSED options which can be tested with the following code:
- Running RadioSED from the command line on a single source and produce output plots: `python radiosed.py -n J213437.6-235535 -p`
- Running RadioSED from the command line on a number of sources stored in an input file: `python radiosed.py -f example_files/input_sources.csv`
- Running RadioSED from the command line on a single source with censored data: `python radiosed.py -n J090331.3+010849 -c example_files/J090331.3+010849_flux_table_censored.csv`

There are more options for saving output data to file, overwriting data from previous runs and using different inputs to identify sources. All of these can be found by calling `python radiosed.py -h`.

The example flux_table files should be used as a template for formatting if you want to use your own flux density data with RadioSED.

## Output
From the command line, you can tell RadioSED to provide plots with the `-p` flag and/or text-based output with the `-w` flag. Plots are saved by default under `output/model_plots`, while the 
summary .csv will appear under `output/data`.

## Citation

If this code is of any use to you in your research, please use the DOI to cite it directly (DOI:10.5281/zenodo.8336847). You can cite either the latest release, or a specific release which your research made use of. We would also appreciate a reference to our forthcoming paper (details coming soon!).

## License

RadioSED is an open source project made available under the GPLv3 license (see LICENCE file). If you require another license, please contact me.
