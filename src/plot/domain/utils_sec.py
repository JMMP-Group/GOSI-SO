#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import os
from os.path import join, isfile, basename, splitext
import glob
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

# =====================================================================================================

def compute_masks(ds_domain, merge=False):
    """
    Compute masks from domain_cfg Dataset.
    If merge=True, merge with the input dataset.
    Parameters
    ----------
    ds_domain: xr.Dataset
        domain_cfg datatset
    add: bool
        if True, merge with ds_domain
    Returns
    -------
    ds_mask: xr.Dataset
        dataset with masks
    """

    # Extract variables
    k = ds_domain["z_c"] + 1
    top_level = ds_domain["top_level"]
    bottom_level = ds_domain["bottom_level"]

    # Page 27 NEMO book.
    # I think there's a typo though.
    # It should be:
    #                  | 0 if k < top_level(i, j)
    # tmask(i, j, k) = | 1 if top_level(i, j) ≤ k ≤ bottom_level(i, j)
    #                  | 0 if k > bottom_level(i, j)
    tmask = xr.where(np.logical_or(k < top_level, k > bottom_level), 0, np.nan)
    tmask = xr.where(np.logical_and(bottom_level >= k, top_level <= k), 1, tmask)
    tmask = tmask.rename("tmask")

    tmask = tmask.transpose("z_c","y_c","x_c")

    # Need to shift and replace last row/colum with tmask
    # umask(i, j, k) = tmask(i, j, k) ∗ tmask(i + 1, j, k)
    umask = tmask.rolling(x_c=2).prod().shift(x_c=-1)
    umask = umask.where(umask.notnull(), tmask)
    umask = umask.rename("umask")

    # vmask(i, j, k) = tmask(i, j, k) ∗ tmask(i, j + 1, k)
    vmask = tmask.rolling(y_c=2).prod().shift(y_c=-1)
    vmask = vmask.where(vmask.notnull(), tmask)
    vmask = vmask.rename("vmask")

    # Return
    masks = xr.merge([tmask, umask, vmask])
    if merge:
        return xr.merge([ds_domain, masks])
    else:
        return masks

# =====================================================================================================

def hvrsn_dst(lon1, lat1, lon2, lat2):
    '''
    This function calculates the great-circle distance in meters between 
    point1 (lon1,lat1) and point2 (lon2,lat2) using the Haversine formula 
    on a spherical earth of radius 6378.137km. 

    The great-circle distance is the shortest distance over the earth's surface.
    ( see http://www.movable-type.co.uk/scripts/latlong.html)

    --------------------------------------------------------------------------------

    If lon2 and lat2 are 2D matrixes, then dist will be a 2D matrix of distances 
    between all the points in the 2D field and point(lon1,lat1).

    If lon1, lat1, lon2 and lat2 are vectors of size N dist wil be a vector of
    size N of distances between each pair of points (lon1(i),lat1(i)) and 
    (lon2(i),lat2(i)), with 0 => i > N .
    '''

    deg2rad = np.pi / 180.
#    ER = 6378.137 * 1000. # Earth Radius in meters
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

# =====================================================================================================

def get_ij_from_lon_lat(lon, lat, glamt, gphit):
    '''
    get_ij_from_lon_lat find closest model grid point i/j from given lat/lon

    Syntax:
      [i, j = get_ij_from_lon_lat(lon, lat, glamt, gphit)
   
    Description:
      returns the i,j model grid position which is closest to the given
      lat/lon point. The model grid is given by 2D lon and lat matrix glamt
      and gphit.
    '''

    dist = hvrsn_dst(lon, lat, glamt, gphit)

    min_dist = np.amin(dist)

    find_min = np.where(dist == min_dist)
    sort_j = np.argsort(find_min[1])

    j_indx = find_min[0][sort_j]
    i_indx = find_min[1][sort_j]

    return j_indx[0], i_indx[0]

# =====================================================================================================

def bresenham_line(x0, x1, y0, y1):
    '''
    point0 = (y0, x0), point1 = (y1, x1)

    It determines the points of an n-dimensional raster that should be 
    selected in order to form a close approximation to a straight line 
    between two points. Taken from the generalised algotihm on

    http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    '''
    steep = abs(y1 - y0) > abs(x1 - x0)
   
    if steep:
       # swap(x0, y0)
       t  = y0
       y0 = x0
       x0 = t
       # swap(x1, y1)    
       t  = y1
       y1 = x1
       x1 = t

    if x0 > x1:
       # swap(x0, x1)
       t  = x1
       x1 = x0
       x0 = t
       # swap(y0, y1)
       t  = y1
       y1 = y0
       y0 = t

    deltax = np.fix(x1 - x0)
    deltay = np.fix(abs(y1 - y0))
    error  = 0.0

    deltaerr = deltay / deltax
    y = y0

    if y0 < y1:
       ystep = 1
    else:
       ystep = -1

    c=0
    pi = np.zeros(shape=[x1-x0+1])
    pj = np.zeros(shape=[x1-x0+1])
    for x in np.arange(x0,x1+1) :
        if steep:
           pi[c]=y
           pj[c]=x
        else:
           pi[c]=x
           pj[c]=y
        error = error + deltaerr
        if error >= 0.5:
           y = y + ystep
           error = error - 1.0
        c += 1

    return pj, pi

# =====================================================================================================

def get_poly_line_ij(points_i, points_j):
    '''
    get_poly_line_ij draw rasterised line between vector-points
    
    Description:
    get_poly_line_ij takes a list of points (specified by 
    pairs of indexes i,j) and draws connecting lines between them 
    using the Bresenham line-drawing algorithm.
    
    Syntax:
    line_i, line_j = get_poly_line_ij(points_i, points_i)
    
    Input:
    points_i, points_j: vectors of equal length of pairs of i, j
                        coordinates that define the line or polyline. The
                        points will be connected in the order they're given
                        in these vectors. 
    Output:
    line_i, line_j: vectors of the same length as the points-vectors
                    giving the i,j coordinates of the points on the
                    rasterised lines. 
    '''
    line_i=[]
    line_j=[]

    line_n=0

    if len(points_i) == 1:
       line_i = points_i
       line_j = points_j
    else:
       for fi in np.arange(len(points_i)-1):
           # start point of line
           i1 = points_i[fi]
           j1 = points_j[fi]
           # end point of line
           i2 = points_i[fi+1]
           j2 = points_j[fi+1]
           # 'draw' line from i1,j1 to i2,j2
           pj, pi = bresenham_line(i1,i2,j1,j2)
           if pi[0] != i1 or pj[0] != j1:
              # beginning of line doesn't match end point, 
              # so we flip both vectors
              pi = np.flipud(pi)
              pj = np.flipud(pj)

           plen = len(pi)

           for PI in np.arange(plen):
               line_n = PI
               if len(line_i) == 0 or line_i[line_n-1] != pi[PI] or line_j[line_n-1] != pj[PI]:
                  line_i.append(int(pi[PI]))
                  line_j.append(int(pj[PI]))


    return line_j, line_i

