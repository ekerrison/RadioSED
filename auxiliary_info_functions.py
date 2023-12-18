#!/usr/bin/python
from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as Coord
from astroquery.simbad import Simbad
import pandas as pd
import numpy as np

class AuxInfo:
    '''
    Helper functions that use astropy and/or astroquery for additional information about sources
    all take in the source coordinates
    '''

    def __init__(self):
        # variables
        self.racs_vizier = "J/other/PASA/38.58/galcut"
        self.gleam_vizier = "VIII/100/gleamegc"
        self.at20g_vizier = "J/MNRAS/434/956/table2"
        self.at20g_radius = 19
        self.gleam_radius = 25
        return


    # function to check compactness in GLEAM source
    def check_gleam_compactness(self, ra=None, dec=None, src_name=None, gleam_viz=None, radius=None):
        """
        Function that checks how compact a source is in GLEAM. Returns the peak/integratd
        flux ratio of the nearest GLEAM source, and the radial distance to that source.
        """
        if gleam_viz is None:
           gleam_viz = self.gleam_vizier

        if radius is None:
            radius = self.gleam_radius

        if src_name is not None and ra is None:
            src_name = src_name.replace("RACS-DR1", "")
            src_name, ra, dec, separation, racs_id = self.resolve_name_generic(src_name)

        # now find nearest racs source
        coords = Coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")
        try:
            res = Vizier.query_region(
                coords, radius=radius * u.arcsec, catalog=gleam_viz
            )[0].to_pandas()
        except IndexError:
            return -1, -1

        # make positions into coord objects
        racs_coord = Coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        gleam_coord = Coord.SkyCoord(
            ra=res["RAJ2000"], dec=res["DEJ2000"], unit=(u.hourangle, u.deg)
        )
        separation = racs_coord.separation(gleam_coord).degree * 60 * 60

        # get flux ratio
        fluxratio = res["Fpwide"] / res["Fintwide"]

        return fluxratio[0], separation[0]


    # function to check compactness in AT20G using Chhetri et al. 2012 catalogue
    def check_at20g_compactness(self, ra=None, dec=None, src_name=None, at20g_viz=None, radius=None):
        """
        Function that checks how compact a source is in Rajan's visibility catalogues for
        AT20G at 20GHz

        returns:
        - boolean for is_compact
        - visibility ratio
        - separation (arcseconds)
        """
        if radius is None:
            radius = self.at20g_radius

        if at20g_viz is None:
            at20g_viz = self.at20g_vizier

        if src_name is not None and ra is None:
            src_name = src_name.replace("RACS-DR1", "")
            src_name, ra, dec, separation, racs_id = self.resolve_name_generic(src_name)
        
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
    def check_racs_compactness(self, src_name: str, racs_viz=None):
        """
        Function that checks how compact a source is in racs. Returns the number of 
        gaussian components it is divided into, as well as the peak/integrated flux ratio.
        """
        if racs_viz is None:
            racs_viz = self.racs_vizier

        src_name = src_name.strip("RACS-DR1 ")
        res = Vizier.query_object(src_name, catalog=racs_viz)[0].to_pandas()
        n_gaus = res["Ng"].values[0]
        fluxratio = res["Fpk"].values[0] / res["Ftot"].values[0]

        return n_gaus, fluxratio

    def check_confusion(self, src_name, radius = 6.5*60, catalog=None):
        """
        Function that determines whether a source is likely suffer from confusion or blending.
        Returns a boolean confusion_flag as well as the number of neighbours
        """
        if catalog is None:
            catalog = self.gleam_vizier

        src_name = src_name.replace("RACS-DR1", "")
        src_name, ra, dec, separation, racs_id = self.resolve_name_generic(src_name)
        
        # now find nearest racs source
        coords = Coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")
        try:
            res = Vizier.query_region(
                coords, radius=radius * u.arcsec, catalog=catalog
            )[0].to_pandas()
        except IndexError:
            return -1, -1

        num_neighbours = res.shape[0]
        confusion_flag = False
        if num_neighbours > 1:
            confusion_flag = True
        return confusion_flag, num_neighbours


    def resolve_name_generic(self, iau_name, racs_viz=None):
        """
        Function that reads in a generic IAU name and returns decimal coordinates for the nearest
        RACS source, with separation
        """
        if racs_viz is None:
            racs_viz = self.racs_vizier

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
        racs_id = res["ID"].values[0]
        ra = res["RAJ2000"].values[0]
        dec = res["DEJ2000"].values[0]

        return racs_name, ra, dec, separation, racs_id


    def resolve_name_racs(self, racs_iau_name):
        """
        Function that reads in the RACS-DR1 name and returns decimal coordinates
        """
        result = Coord.get_icrs_coordinates(racs_iau_name)
        return result.ra.deg, result.dec.deg


    def racs_id_to_name(self, racs_id):
        """
        Function that converts the RACS ID (beginning RACS_) to the RACS IAU
        name (RACS-DR1 Jxxxxxx+xxxxxx)
        """

        res = Vizier.query_constraints(catalog=racs_vizier, ID="={}".format(racs_id))[
            0
        ].to_pandas()
        racs_iau_name = res["RACS-DR1"].values[0]
        return racs_iau_name


    def find_racs_src(self, ra, dec, racs_id=None):
        return racs_iau_name


# tests
if __name__ == "__main__":
    test_name = "RACS_2004+00A_2608"
    info = AuxInfo()
    # test_name = 'J200826.5+001049'
    racs_vizier = "J/other/PASA/38.58/galcut"
    gleam_vizier = "VIII/100/gleamegc"
    res = Vizier.query_object(test_name, catalog=racs_vizier)
    res = Vizier.query_object(test_name, catalog=racs_vizier)
    res = info.racs_id_to_name(test_name)
    print(res)
    res = info.check_confusion(res)
    print(res)

    test_name = 'J000311-544516'
    racs_name, ra, dec, sep, racs_id = info.resolve_name_generic(test_name)
    print(racs_name, ra, dec, sep)
    res = info.check_gleam_compactness(ra=ra, dec=dec)
    print(res)

    res = Coord.get_icrs_coordinates("RACS-DR1 J200826.5+001049")
    res = Vizier.query_region(
        "RACS-DR1 J200826.5+001049", radius=1 * u.arcmin, catalog=racs_vizier
    )
    print(res[0].to_pandas())
