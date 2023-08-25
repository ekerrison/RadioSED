# RaiSED

## A Bayesian Nested Sampling approach to radio SED fitting for AGN.

This package uses nested sampling ([Skilling 2004](https://doi.org/10.1063/1.1835238)) to perform a Bayesian analysis of radio SEDs constructed from radio flux density measurements
obtained as part of large area surveys (or in some limited cases, as part of targeted followup campaigns). Users can make use of a pre-defined set of models and surveys from which to draw
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

More documentation coming soon!

## Usage

To see how RaiSED is run, please take a look at the `run_raised.py` script. This can be used as the main script from which you run fitting on your own sources, or it can
be modified to suit your needs.

If this code is of any use to you in your research, we would appreciate you reference our forthcoming paper.
