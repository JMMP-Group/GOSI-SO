#!/usr/bin/env python

from typing import Tuple
import numpy as np
import netCDF4 as nc4
from scipy.ndimage import gaussian_filter
import xarray as xr
from xarray import Dataset, DataArray 

import matplotlib.pyplot as plt

#=======================================================================================
def read_envInfo(filepath):
    '''
    This function reads the parameters which will be used
    to generate the model grid envelopes.

    If the file is not in the right format and does not contain
    all the needed infos an error message is returned.

    filepath: string
    '''

    import importlib.machinery as imp

    loader  = imp.SourceFileLoader(filepath,filepath)
    envInfo = loader.load_module()

    attr = ['bathyFile', 'hgridFile', 'zgridFile', 'e_min_ofs', 
            'e_max_dep', 'e_loc_vel', 'e_loc_var', 'e_loc_vmx', 
            'e_loc_rmx', 'e_loc_hal', 'e_glo_rmx', 'e_tap_equ']

    errmsg = False
    for a in attr:
        if not hasattr(envInfo,a):
           errmsg = True
           attrerr = a
        else:
           if getattr(envInfo,a) == "":
              errmsg = True
              attrerr = a

    if errmsg: 
       raise AttributeError(
             'Attribute ' + attrerr + ' is missing'
       )

    return envInfo

#=======================================================================================
def calc_zenv(bathy, surf, offset, max_dep):
    """
    Constructs an envelop bathymetry 
    for the sigma levels definition.
       
    INPUT:
    *) bathy: the bathymetry field which will be 
              used to compute the envelope.
    *) surf:  2D field used as upper surface to model 
              the envelope
    *) offset: offset to be added to surf to calc 
               the minimum depth of the envelope.

    *) max_depth : maximum depth of the envelope.
    """

    zenv   = bathy.copy()
    env_up = surf.copy()

    # Set minimum depth of the envelope
    env_up += offset
    zenv = np.maximum(zenv, env_up)

    if max_dep < 0:
       max_dep *= -1.
       msk_val = -99
       msk = zenv.where(zenv > max_dep, msk_val)
       #msk.plot(vmin=-99,vmax=1)
       #plt.show()
       # TO ADD a check if the isobath defines a closed polygon
       open_ocean_pnt = dict(lon=-43.53, lat=32.01)
       j_start, i_start = get_ij_from_lon_lat(open_ocean_pnt['lon'], 
                                              open_ocean_pnt['lat'], 
                                              bathy.nav_lon, 
                                              bathy.nav_lat
                          )
       msk_fill = floodfill(msk, j_start, i_start, msk_val, -9999.)
       msk.data = msk_fill
       msk = xr.where(msk==-9999.,0,1)
    else:
       msk = bathy*0. + 1

    # Set maximum depth of the envelope
    zenv = zenv.where(msk==1,np.minimum(zenv, max_dep))

    # Set minimum depth of the envelope
    #env_up += offset
    #zenv = np.maximum(zenv, env_up)

    return zenv

#=======================================================================================
def calc_rmax(depth: DataArray) -> DataArray:
    """
    Calculate rmax: measure of steepness
    This function returns the slope paramater field

    r = abs(Hb - Ha) / (Ha + Hb)

    where Ha and Hb are the depths of adjacent grid cells (Mellor et al 1998).

    Reference:
    *) Mellor, Oey & Ezer, J Atm. Oce. Tech. 15(5):1122-1131, 1998.

    Parameters
    ----------
    depth: DataArray
        Bottom depth (units: m).

    Returns
    -------
    DataArray
        2D slope parameter (units: None)

    Notes
    -----
    This function uses a "conservative approach" and rmax is overestimated.
    rmax at T points is the maximum rmax estimated at any adjacent U/V point.
    """
    # Mask land
    depth = depth.where(depth > 0)

    # Loop over x and y
    both_rmax = []
    for dim in depth.dims:

        # Compute rmax
        rolled = depth.rolling({dim: 2}).construct("window_dim")
        diff = rolled.diff("window_dim").squeeze("window_dim")
        rmax = np.abs(diff) / rolled.sum("window_dim")

        # Construct dimension with velocity points adjacent to any T point
        # We need to shift as we rolled twice
        rmax = rmax.rolling({dim: 2}).construct("vel_points")
        rmax = rmax.shift({dim: -1})

        both_rmax.append(rmax)

    # Find maximum rmax at adjacent U/V points
    rmax = xr.concat(both_rmax, "vel_points")
    rmax = rmax.max("vel_points", skipna=True)

    # Mask halo points
    for dim in rmax.dims:
        rmax[{dim: [0, -1]}] = 0

    return rmax.fillna(0)

#=======================================================================================
def smooth_MB06(
    depth: DataArray,
    rmax: float,
    tol: float = 1.0e-8,
    max_iter: int = 10_000,
) -> DataArray:
    """
    Direct iterative method of Martinho and Batteen (2006) consistent
    with NEMO implementation.

    The algorithm ensures that

                H_ij - H_n
                ---------- < rmax
                H_ij + H_n

    where H_ij is the depth at some point (i,j) and H_n is the
    neighbouring depth in the east, west, south or north direction.

    Reference:
    *) Martinho & Batteen, Oce. Mod. 13(2):166-175, 2006.

    Parameters
    ----------
    depth: DataArray
        Bottom depth.
    rmax: float
        Maximum slope parameter allowed
    tol: float, default = 1.0e-8
        Tolerance for the iterative method
    max_iter: int, default = 10000
        Maximum number of iterations

    Returns
    -------
    DataArray
        Smooth version of the bottom topography with
        a maximum slope parameter < rmax.
    """

    # Set scaling factor used for smoothing
    zrfact = (1.0 - rmax) / (1.0 + rmax)

    # Initialize envelope bathymetry
    zenv = depth

    for _ in range(max_iter):

        # Initialize lists of DataArrays to concatenate
        all_ztmp = []
        all_zr = []
        for dim in zenv.dims:

            # Shifted arrays
            zenv_m1 = zenv.shift({dim: -1})
            zenv_p1 = zenv.shift({dim: +1})

            # Compute zr
            zr = (zenv_m1 - zenv) / (zenv_m1 + zenv)
            zr = zr.where((zenv > 0) & (zenv_m1 > 0), 0)
            for dim_name in zenv.dims:
                zr[{dim_name: -1}] = 0
            all_zr += [zr]

            # Compute ztmp
            zr_p1 = zr.shift({dim: +1})
            all_ztmp += [zenv.where(zr <= rmax, zenv_m1 * zrfact)]
            all_ztmp += [zenv.where(zr_p1 >= -rmax, zenv_p1 * zrfact)]

        # Update envelope bathymetry
        zenv = xr.concat([zenv] + all_ztmp, "dummy_dim").max("dummy_dim")

        # Check target rmax
        zr = xr.concat(all_zr, "dummy_dim")
        #print(np.nanmax(np.abs(zr)))
        if ((np.abs(zr) - rmax) <= tol).all():
            return zenv

    raise ValueError(
        "Iterative method did NOT converge."
        " You might want to increase the number of iterations and/or the tolerance."
        " The maximum slope parameter rmax is " + str(np.nanmax(np.abs(zr)))
    )

#=======================================================================================
def e3_to_dep(e3W: DataArray, e3T: DataArray) -> Tuple[DataArray, ...]:

    gdepT = xr.full_like(e3T, None, dtype=np.double).rename('gdepT')
    gdepW = xr.full_like(e3W, None, dtype=np.double).rename('gdepW')

    gdepW[{"z":0}] = 0.0
    gdepT[{"z":0}] = 0.5 * e3W[{"z":0}]
    for k in range(1, e3W.sizes["z"]):
        gdepW[{"z":k}] = gdepW[{"z":k-1}] + e3T[{"z":k-1}]
        gdepT[{"z":k}] = gdepT[{"z":k-1}] + e3W[{"z":k}]

    return tuple([gdepW, gdepT])

#=======================================================================================
def msg_info(message,main=False):

    if main:
       print('')
       print('='*76)
       print(' '*11 + message)
       print('='*76)
    else:
       print(message)
    print('')

#=======================================================================================
def get_ij_from_lon_lat(LON, LAT, lon, lat):
    '''
    This function finds the closest model 
    grid point i/j to a given lat/lon.

    Syntax:
    i, j = get_ij_from_lon_lat(LON, LAT, lon, lat)
   
    LON, LAT: target longitude and latitude
    lon, lat: 2D arrays of model grid's longitude and latidtude
    '''

    dist = hvrsn_dst(LON, LAT, lon, lat)

    min_dist = np.amin(dist)

    find_min = np.where(dist == min_dist)
    sort_j = np.argsort(find_min[1])

    j_indx = find_min[0][sort_j]
    i_indx = find_min[1][sort_j]

    return j_indx[0], i_indx[0]

#=======================================================================================
def hvrsn_dst(lon1, lat1, lon2, lat2):
    '''
    This function calculates the great-circle distance in meters between 
    point1 (lon1,lat1) and point2 (lon2,lat2) using the Haversine formula 
    on a spherical earth of radius 6372.8 km. 

    The great-circle distance is the shortest distance over the earth's surface.
    ( see http://www.movable-type.co.uk/scripts/latlong.html)

    If lon2 and lat2 are 2D matrixes, then dist will be a 2D matrix of distances 
    between all the points in the 2D field and point(lon1,lat1).

    If lon1, lat1, lon2 and lat2 are vectors of size N dist wil be a vector of
    size N of distances between each pair of points (lon1(i),lat1(i)) and 
    (lon2(i),lat2(i)), with 0 => i > N .
    '''
    deg2rad = np.pi / 180.
    ER = 6372.8 * 1000. # Earth Radius in meters

    dlon = np.multiply(deg2rad, (lon2 - lon1))
    dlat = np.multiply(deg2rad, (lat2 - lat1))

    lat1 = np.multiply(deg2rad, lat1)
    lat2 = np.multiply(deg2rad, lat2)

    # Computing the square of half the chord length between the points:
    a = np.power(np.sin(np.divide(dlat, 2.)),2) + \
        np.multiply(np.multiply(np.cos(lat1),np.cos(lat2)),np.power(np.sin(np.divide(dlon, 2.)),2))

    # Computing the angular distance in radians between the points
    angle = np.multiply(2., np.arctan2(np.sqrt(a), np.sqrt(1. -a)))

    # Computing the distance 
    dist = np.multiply(ER, angle)

    return dist

#=======================================================================================
def floodfill(field,j,i,checkValue,newValue):
    '''
    This is a modified version of the original algorithm:

    1) checkValue is the value we do not want to change,
       i.e. is the value identifying the boundaries of the 
       region we want to flood.
    2) newValue is the new value we want for points whose initial value
       is not checkValue and is not newValue.
       N.B. if a point with initial value = to newValue is met, then the
            flooding stops. 

    Example:

    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 3, 2, 1, 5, 6, 9, 0],
                  [0, 0, 8, 9, 0, 0, 0, 4, 0],
                  [0, 0, 8, 9, 7, 2, 3, 0, 0],
                  [0, 0, 4, 4, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0]])
   
    j_start = 3
    i_start = 4
    b = com.floodfill(a,j_start,i_start,0,2)
 
    b = array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 2, 2, 1, 5, 6, 9, 0],
               [0, 0, 2, 2, 0, 0, 0, 4, 0],
               [0, 0, 2, 2, 2, 2, 3, 0, 0],
               [0, 0, 2, 2, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    '''
    Field = np.copy(field)

    theStack = [ (j, i) ]

    while len(theStack) > 0:
          try:
              j, i = theStack.pop()
              if Field[j,i] == checkValue:
                 continue
              if Field[j,i] == newValue:
                 continue
              Field[j,i] = newValue
              theStack.append( (j, i + 1) )  # right
              theStack.append( (j, i - 1) )  # left
              theStack.append( (j + 1, i) )  # down
              theStack.append( (j - 1, i) )  # up
          except IndexError:
              continue # bounds reached

    return Field
