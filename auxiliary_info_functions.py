#!/usr/bin/python
from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as Coord
from astroquery.simbad import Simbad
import pandas as pd
import numpy as np

# helper functions that use astropy and/or astroquery for additional information about sources
# all take in the source coordinates

# variables
racs_vizier = "J/other/PASA/38.58/galcut"
gleam_vizier = "VIII/100/gleamegc"
at20g_vizier = "J/MNRAS/434/956/table2"


# function to check compactness in GLEAM source
def check_gleam_compactness(ra, dec, gleam_viz=gleam_vizier, radius=25):
    """
    Function that checks how compact a source is in GLEAM
    """

    # now find nearest racs source
    coords = Coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")
    res = Vizier.query_region(coords, radius=radius * u.arcsec, catalog=gleam_vizier)
    print(res)
    try:
        res = Vizier.query_region(
            coords, radius=radius * u.arcsec, catalog=gleam_vizier
        )[0].to_pandas()
    except IndexError:
        return -1, -1, -1
    print(res.columns)

    # make positions into coord objects
    racs_coord = Coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    at20g_coord = Coord.SkyCoord(
        ra=res["RAJ2000"], dec=res["DEJ2000"], unit=(u.hourangle, u.deg)
    )
    separation = racs_coord.separation(at20g_coord).degree * 60 * 60

    # get flux ratio
    fluxratio = res["Fpwide"] / res["Fintwide"]

    return fluxratio[0], separation[0]


# function to check compactness in AT20G using Chhetri et al. 2012 catalogue
def check_at20g_compactness(ra, dec, at20g_viz=at20g_vizier, radius=19):
    """
    Function that checks how compact a source is in Rajan's visibility catalogues for
    AT20G at 20GHz

    return:
     - boolean for is_compact
     - visibility ratio
     - separation (arcseconds)
    """
    # now find nearest racs source
    coords = Coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")
    try:
        res = Vizier.query_region(coords, radius=radius * u.arcsec, catalog=at20g_viz)[
            0
        ].to_pandas()
    except IndexError:
        return -1, -1, -1

    # make positions into coord objects
    racs_coord = Coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    at20g_coord = Coord.SkyCoord(
        ra=res["RAJ2000"], dec=res["DEJ2000"], unit=(u.hourangle, u.deg)
    )
    separation = racs_coord.separation(at20g_coord).degree * 60 * 60

    vis1 = float(res["Vis1"].values[0])
    vis2 = float(res["Vis2"].values[0])

    visibility = np.nanmin([vis1, vis2])
    is_compact = (
        visibility > 0.86
    )  # from Chhetri et al. 2012, MNRAS 422, 2274 (section 3)

    return is_compact, visibility, separation[0]


# function to check compactness in RACS
def check_racs_compactness(racs_iau_name: str, racs_viz=racs_vizier):
    """
    Function that checks how compact a source is in racs
    """
    racs_iau_name = racs_iau_name.strip("RACS-DR1 ")
    res = Vizier.query_object(racs_iau_name, catalog=racs_viz)[0].to_pandas()
    n_gaus = res["Ng"].values[0]
    fluxratio = res["Fpk"].values[0] / res["Ftot"].values[0]

    return n_gaus, fluxratio


def resolve_name_generic(iau_name, racs_viz=racs_vizier):
    """
    Function that reads in a generic IAU name and returns decimal coordinates for the nearest
    RACS source, with separation
    """
    result = Coord.get_icrs_coordinates(iau_name)
    ra, dec = result.ra.deg, result.dec.deg

    # now find nearest racs source
    try:
        res = Vizier.query_object(iau_name, catalog=racs_viz)[0].to_pandas()
    except IndexError as e:
        print("No RACS source found within 1 arcminute of {}".format(iau_name))
        return -1, -1, -1, -1

    # separation = np.sqrt((res['RAJ2000'] - ra)**2 + (res['DEJ2000'] - dec)**2)*(60*60)
    racs_coord = Coord.SkyCoord(
        ra=res["RAJ2000"].values[0] * u.deg, dec=res["DEJ2000"].values[0] * u.deg
    )

    #add coord objects to calculate separation
    res['coords'] = res.apply(lambda x: Coord.SkyCoord(
        ra=x.RAJ2000 * u.deg, dec=x.DEJ2000 * u.deg),
        axis = 1)

    res['sep'] = res.apply(lambda x: result.separation(x.coords), axis = 1)
    
    #return closest source
    res = res.sort_values(by = 'sep')

    separation = separation = res['sep'].values[0]

    if separation.deg * 60 * 60 > 60:
        raise IndexError(
            "No RACS source found within 1 arcminute of {}".format(iau_name)
        )

    racs_name = res["RACS-DR1"].values[0]
    ra = res["RAJ2000"].values[0]
    dec = res["DEJ2000"].values[0]

    return racs_name, ra, dec, separation


def resolve_name_racs(racs_iau_name):
    """
    Function that reads in the RACS-DR1 name and returns decimal coordinates
    """
    result = Coord.get_icrs_coordinates(racs_iau_name)
    return result.ra.deg, result.dec.deg


def racs_id_to_name(racs_id):
    """
    Function that converts the RACS ID (beginning RACS_) to the RACS IAU
    name (RACS-DR1 Jxxxxxx+xxxxxx)
    """

    res = Vizier.query_constraints(catalog=racs_vizier, ID="={}".format(racs_id))[
        0
    ].to_pandas()
    racs_iau_name = res["RACS-DR1"].values[0]
    return racs_iau_name


def find_racs_src(ra, dec, racs_id=None):
    return racs_iau_name


# tests
if __name__ == "__main__":
    test_name = "RACS_2004+00A_2608"
    # test_name = 'J200826.5+001049'
    # test_name = 'J000311-544516'
    res = Vizier.query_object(test_name, catalog=racs_vizier)
    res = Vizier.query_object(test_name, catalog=racs_vizier)
    res = racs_id_to_name(test_name)
    print(res)
    exit()

    racs_name, ra, dec, sep = resolve_name_generic(test_name)
    print(racs_name, ra, dec, sep)
    res = check_gleam_compactness(ra=ra, dec=dec)
    print(res)
    exit()
    res = Coord.get_icrs_coordinates("RACS-DR1 J200826.5+001049")
    res = Vizier.query_region(
        "RACS-DR1 J200826.5+001049", radius=1 * u.arcmin, catalog=racs_vizier
    )
    print(res[0].to_pandas())
