# RadioSED

## A Bayesian Nested Sampling approach to radio SED fitting for AGN.

This package uses nested sampling ([Skilling 2004](https://doi.org/10.1063/1.1835238)) to perform a Bayesian analysis of radio SEDs constructed from radio flux density measurements
obtained as part of large area surveys (or in some limited cases, as part of targeted followup campaigns). It is essentially a wrapper around [Bilby](https://lscsoft.docs.ligo.org/bilby/#),
 the Bayesian inference library developed for Gravitational-wave astronomy, and it makes use of [Dynesty](https://dynesty.readthedocs.io/en/latest/index.html) to perform the sampling steps,
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

Where the 'survey quickname' is some easily identifiable name (e.g. RACS) and the Refcode is the bibcode for the relevant survey paper.

More documentation coming soon!

## Usage

To run RadioSED you will obviously need both [Bilby](https://lscsoft.docs.ligo.org/bilby/#) and [Dynesty](https://dynesty.readthedocs.io/en/latest/index.html), but additional 
package requirements can be found in `requirements.txt` (for pip-compatible formatting) or `requirements_conda.txt` (for anaconda-style formatting).  
If you are running anaconda, once you clone this repo you can run `conda create --name radiosed --file requirements_conda.txt` from within the folder to create a new environment
with all the packages required to run RadioSED.

You will also need to have [Git Large File Storage](https://git-lfs.com/) installed in order to download the requisite files (you may need to do another `git pull` after installing the LFS
in order to pull these files across). The necessary files are located within the `data\` subdirectory.

If you would like to make use of the pre-determined crossmatches that come with RadioSED, you will need to initialise the DataParser with option `use_local=True`.

To see how RadioSED is run, please take a look at the `run_radiosed.py` script. This can be used as the main script from which you run fitting on your own sources, or it can
be modified to suit your needs.

If this code is of any use to you in your research, we would appreciate a reference to our forthcoming paper.

## License

RadioSED is an open source project made available under the GPLv3 license (see LICENCE file). If you require another license, please contact me.
