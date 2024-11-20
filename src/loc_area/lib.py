#!/usr/bin/env python

from typing import Tuple
import numpy as np
import netCDF4 as nc4
import xarray as xr
import scipy.spatial as sp
from xarray import Dataset, DataArray 
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

#=======================================================================================
def read_locInfo(filepath):
    '''
    This function reads the parameters which will be used
    to generate the localisation areas.

    If the file is not in the right format and does not contain
    all the needed infos an error message is returned.

    filepath: string
    '''

    import importlib.machinery as imp

    loader  = imp.SourceFileLoader(filepath,filepath)
    locInfo = loader.load_module()

    attr = ['bathyFile', 'hgridFile', 'loc_isos' , 'loc_area' , 
            's2z_factr', 's2z_sigma', 's2z_stenc', 's2z_itera',
            's2z_wghts'
           ]

    errmsg = False
    for a in attr:
        if not hasattr(locInfo,a):
           errmsg = True
           attrerr = a
        else:
           if getattr(locInfo,a) == "":
              errmsg = True
              attrerr = a

    if errmsg: 
       raise AttributeError(
             'Attribute ' + attrerr + ' is missing'
       )

    return locInfo

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
def find_bdy_JI(mask: np.ndarray) -> np.ndarray:
    """
    Identify boundary points of a mask array.
    
    This function is used to identify and return the boundary points of a mask array.
    The mask array is defined using 0|1 and the indices return are the points where
    the value 1 is adjacent to 0.

    Parameters
    ----------
    mask: np.ndarray
        An array of dimension 2 containing zeros or ones, 
        which define a masked region

    Returns
    -------
    tji[uniqind]: np.ndarray
        The indices of the boundary between the masked region

    Notes
    -----
    """
    
    # Determine size of input array
    nJ, nI = mask.shape
    
    # Bearings for overlays
    _NORTH = [1, -1, 1, -1, 2, None, 1, -1]
    _SOUTH = [1, -1, 1, -1, None, -2, 1, -1]
    _EAST  = [1, -1, 1, -1, 1, -1, 2, None]
    _WEST  = [1, -1, 1, -1, 1, -1, None, -2]
    
    # Create padded array for overlays
    msk = np.pad(mask, ((1, 1), (1, 1)), "constant", constant_values=(-999))
        
    # Create index arrays of J and I coords
    jgrid, igrid = np.meshgrid(np.arange(nJ), np.arange(nI), indexing='ij')

    SBj, SBi = find_bdy(jgrid, igrid, msk, _SOUTH)
    NBj, NBi = find_bdy(jgrid, igrid, msk, _NORTH)
    EBj, EBi = find_bdy(jgrid, igrid, msk, _EAST)
    WBj, WBi = find_bdy(jgrid, igrid, msk, _WEST)    
        
    # Create a 2D array index for the points that are on border
    tji = np.column_stack( (np.concatenate((SBj, NBj, WBj, EBj)), 
                            np.concatenate((SBi, NBi, WBi, EBi))) )
    uniqind = unique_rows(tji)
    
    return tji[uniqind]
        
def unique_rows(ind_arr: np.ndarray) -> np.ndarray:
    """
    Remove any duplicates from an index array.
    
    This function is used to remove duplicated boundary indices
    generated in function find_bdy.

    Parameters
    ----------
    ind_arr: np.ndarray
        nx2 array of J, I indices defining the boundary of a
        mask region.

    Returns
    -------
    indx[indx != -1]: np.ndarray
        The indices unique values.

    Notes
    -----
    """
    
    # Determine input shape
    shp = np.shape(ind_arr)
    
    # Check
    if (len(shp) > 2) or (shp[0] == 0) or (shp[1] == 0):
        print("Warning: Shape of expected 2D array:", shp)
    
    # Create a sorted list of indices
    ind_list = ind_arr.tolist()
    ind_sort = []
    indx = list(zip(*sorted([(val, i) for i, val in enumerate(ind_list)])))[1]
    indx = np.array(indx)
    
    for i in indx:
        ind_sort.append(ind_list[i])
    
    del ind_list
    
    for i, x in enumerate(ind_sort):
        if x == ind_sort[i - 1]:
            indx[i] = -1
    
    # If all the rows are identical, set the first as the unique row
    if ind_sort[0] == ind_sort[-1]:
        indx[0] = 0

    return indx[indx != -1]

def find_bdy(
    jgrid: np.ndarray, 
    igrid: np.ndarray, 
    mask: np.ndarray,
    brg: list,
    ) -> tuple[ np.ndarray, np.ndarray] :
    """
    Identify boundary points of a mask array.
    
    This function is used to identify and return the boundary points of a mask array.
    The mask array is defined using 0|1 and the indices return are the points where
    the value 1 is adjacent to 0.

    Parameters
    ----------
    jgrid: np.ndarray
        An array of dimension 2 containing J indices of the masked region.
    igrid: np.ndarray
        An array of dimension 2 containing I indices of the masked region.
    mask: np.ndarray
        An array of dimension 2 containing zeros or ones, 
        which define a masked region.
    brg: list
        List of the bearing overlays.

    Returns
    -------
    bdy_J: np.ndarray
        The J indices of the boundary between the masked region
    bdy_I: np.ndarray
        The I indices of the boundary between the masked region

    Notes
    -----
    """
    
    # Subtract matrices to find boundaries, set to True
    mat1 = mask[brg[2] : brg[3], brg[0] : brg[1]]
    mat2 = mask[brg[6] : brg[7], brg[4] : brg[5]]

    overlay = np.subtract(mat1, mat2)
    
    # Create boolean array of bdy points in overlay
    bool_arr = overlay == 1

    # Index I or J to find BDY points
    bdy_J = jgrid[bool_arr]
    bdy_I = igrid[bool_arr]

    return bdy_J, bdy_I

#=======================================================================================
def weighting_dist(msk_zones: np.ndarray, a: float) -> np.ndarray:
    """
    Create weights for the transition zone between coordinates.

    This function is used to create a weights array to allow the
    smooth transition from one vertical coordinate system to another.

    Parameters
    ----------
    mask_zones: np.ndarray
        An array of dimension 2 containing zeros, ones, twos. Zeros
        indicate global vertical grid zone, Twos the zone for the
        localised coordinates and ones the transistion zone between
        the two.
    a: float
        A tuning parameter for the transition zone.

    Returns
    -------
    weights: np.ndarray
        An array of weights to apply to the transition zone.

    Notes
    -----
    """

    # Determine size of input array
    nJ, nI = msk_zones.shape

    # Area of localised vertical coordinates
    msk_inner = np.zeros((nJ, nI)).astype(int)
    msk_inner[msk_zones==2] = 1

    # Area of global vertical coordinates
    msk_outer = np.zeros((nJ, nI)).astype(int)
    msk_outer[msk_zones==0] = 1

    # Find boundary points between the zones
    bdy_inner = find_bdy_JI(msk_inner)
    bdy_outer = find_bdy_JI(msk_outer)

    # Initialise the weights array
    weights = np.zeros((nJ,nI))
    weights[msk_outer == 1] = 0.0 # where we only want global vertical coordinates
    weights[msk_inner == 1] = 1.0 # where we only want localised vertical coordinates

    # Create x and y indices for reference
    x    = np.zeros_like(msk_zones); x += np.arange(nI)
    y    = np.zeros_like(msk_zones); y += np.arange(nJ)[:,np.newaxis]

    ind = msk_inner+msk_outer==0 # identify only valid points
    trans_pts = list(zip(y[ind], x[ind]))

    # Create source tree in which to identify distance of points in transition zone
    # to the closest point on the inner boundary
    source_tree = None

    source_tree = sp.cKDTree(
                bdy_inner,
                balanced_tree=False,
                compact_nodes=False,
                )

    # Return distance of closest point on inner boundary
    dist_inner, nn_id = source_tree.query(trans_pts, k=1)

    # Create source tree in which to identify distance of points in transition zone
    # to the closest point on the inner boundary
    source_tree = None

    source_tree = sp.cKDTree(
                bdy_outer,
                balanced_tree=False,
                compact_nodes=False,
                )

    # Return distance of closest point on inner boundary
    dist_outer, nn_id = source_tree.query(trans_pts, k=1)

    weights[ind] = 0.5*(np.tanh(a*(2*dist_outer/(dist_inner + dist_outer)-1))/np.tanh(a)+1)

    return weights

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

#===================================================================================================
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

#=======================================================================================
def get_poly_area_ij(points_i, points_j, a_ji):
    '''
    Syntax:
    area_j, area_i = get_poly_area_ij(points_i, points_j, a_ji)

    Input:
    points_i, points_j: j,i indexes of the the points
                        defining the polygon
    a_ji: shape (i.e., (nj,ni)) of the 2d matrix from which points_i 
          and points_j are selected
    '''
    [jpj, jpi] = a_ji
    pnt_i = np.array(points_i)
    pnt_j = np.array(points_j)

    if (pnt_i[0] == pnt_i[-1]) and (pnt_i[0] == pnt_i[-1]):
        # polygon is already closed
        i_indx = np.copy(pnt_i)
        j_indx = np.copy(pnt_j)
    else:
        # close polygon
        i_indx = np.append( pnt_i, pnt_i[0])
        j_indx = np.append( pnt_j, pnt_j[0])

    [bound_j, bound_i] = get_poly_line_ij(i_indx, j_indx)

    mask = np.zeros(shape=(jpj,jpi))
    for n in range(len(bound_i)):
        i         = bound_i[n]
        j         = bound_j[n]
        mask[j,i] = 1
    corners_j = [ 0,   0, jpj, jpj ]
    corners_i = [ 0, jpi,   0, jpi ]

    for n in range(4):
        j = corners_j[n]
        i = corners_i[n]
        if mask[j,i] == 0:
           fill_i = i
           fill_j = j
           break

    mask_filled = floodfill(mask,fill_j,fill_i,1,1)
    for n in range(len(bound_i)):
        j                = bound_j[n]
        i                = bound_i[n]
        mask_filled[j,i] = 0

    # the points that are still 0 are within the required area
    [area_j, area_i] = np.where(mask_filled == 0);

    return area_j, area_i

#=======================================================================================
def generate_loc_area(bathy, isosurf):

    if isosurf >= 0: # isobath

       s2z_itera=3
       s2z_sigma = 3
       wrk = bathy.copy()
       for nit in range(s2z_itera):
          bathy_gauss = gaussian_filter(wrk, sigma=s2z_sigma, truncate=(2.*s2z_sigma))
          wrk = bathy_gauss.copy()
       #bathy condition
       s_msk   = bathy.where(bathy_gauss<isosurf, -1)
       s_msk.plot()
       plt.show()
      
       #latitude condition
       cond1=s_msk.nav_lat<-48.3
       s_msk = s_msk.where(cond1,-1)
       s_msk.plot()
       plt.show()
       
       #latitude Drake passage
       cond1=np.logical_or(s_msk.nav_lon <-55,s_msk.nav_lon > -32.25)
       cond2=s_msk.nav_lat<-56.1
       s_msk = s_msk.where(np.logical_or(cond1,cond2),-1) 
       s_msk.plot()
       plt.show()


       #remove unwanted area
       cond1=np.logical_or(s_msk.nav_lon <-65.7,s_msk.nav_lon > -62.75)
       cond2=s_msk.nav_lat<-61.3
       s_msk = s_msk.where(np.logical_or(cond1,cond2),-1)
       s_msk.plot()
       plt.show()
       
       #remove unwanted area
       cond1=np.logical_or(s_msk.nav_lon <80.2,s_msk.nav_lon > 87.5)
       cond2=s_msk.nav_lat<-63.8
       s_msk = s_msk.where(np.logical_or(cond1,cond2),-1)
       s_msk.plot()
       plt.show()
       
       s_msk   = s_msk.where(s_msk==-1, 1)
       s_msk   = s_msk.where(s_msk==1, 0)
       s_msk.plot()
       plt.show()

       # MASK FOR ANTARCTICA
       lsm_msk = s_msk.copy()
       lsm_msk[0,:] = -1
       i_start = 680
       j_start = 90
       aa_msk_fill = floodfill(lsm_msk.data, j_start, i_start, 0, -1.)

       msk = s_msk.data

       # only ANTARCTICA
       msk[aa_msk_fill>=0] = 0
       s_msk.data = msk

    elif isosurf < 0: # latitude in the southern emisphere
       s_msk   = xr.where(bathy.nav_lat<isosurf,1, 0)

    s_msk.plot()
    plt.show()

    return s_msk

