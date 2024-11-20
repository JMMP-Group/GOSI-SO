#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import os
from os.path import join, isfile, basename, splitext
import glob
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as feature
from utils_sec import *

# ==============================================================================
# Input parameters

# 1. INPUT FILES

maxdep = 5000.

bathy_file = '/net/home/h01/dbruciaf/mod_dev/GOSI-SO/src/envelopes/bathymetry.loc_area-antarctica.dep3200_sig3_stn9_itr1.MEs_4env_3200_r12_r12.nc'
coord_file = '/data/users/dbruciaf/GOSI10_input_files/p1.0/domcfg_eORCA025_v3.1_r42.nc'

# Loading domain geometry
ds_bat  = xr.open_dataset(bathy_file).squeeze()
ds_cor  = xr.open_dataset(coord_file).squeeze()

LON = ds_cor.glamt.data
LAT = ds_cor.gphit.data

# Weddel Sea
# a) ADELIE section
#   Thompson & Heywood 2008 (https://doi.org/10.1016/j.dsr.2008.06.001) 
#   Thompson et al. 2009 (https://doi.org/10.1175/2008JPO3995.1)
#   Thompson et al. 2018 (https://doi.org/10.1029/2018RG000624) 
#sec_lon1 = [-54.07, -47.49]
#sec_lat1 = [-63.16, -64.18]

# b) Sections at longitudes 32W, and 55W (Fig. 8 b and c)
#    Nicholls et al. 2009 (10.1029/2007RG000250)
#sec_lon2 = [-32.0, -32.0]
#sec_lat2 = [-75.0, -73.8]
#sec_lon3 = [-55.0, -55.0]
#sec_lat3 = [-70.9, -73.1]

# c) Section F1-F4
#    Foldvik et al 2004 (10.1029/2003JC002008) (Fig. 6)
#    Daae et al 2019 (10.1175/JPO-D-18-0093.1) (Fig. 2)
#sec_lon4 = [-36.80, -36.60, -35.70]
#sec_lat4 = [-74.60, -74.52, -74.15]

# Ross Sea
# a) Section NBP0402_s52-s64
#    Gordon et al. 2015 (10.1002/2015GL064457)
#    http://ocp.ldeo.columbia.edu/res/div/ocp/projects/anslope/nbp0402/cruisemap.pdf
#    http://ocp.ldeo.columbia.edu/res/div/ocp/projects/anslope/nbp0402/stationtable.html
#    stations: 52, 64
#sec_lon5 = [171.33, 172.83]
#sec_lat5 = [-71.56, -71.40]

# Ice-shelf cavities
sec_lon1  = [154.81, 174.99, -174. ]
sec_lat1  = [-80.39, -77.96, -67.30]
sec_lon2  = [172.39, -162. ]
sec_lat2  = [-84.19, -71.09]
sec_lon3  = [-106.13, -108.50]
sec_lat3  = [-75.92 , -67.58]
sec_lon4  = [-83.89, -43. ]
sec_lat4  = [-79.92, -67.96]
sec_lon5  = [-64.30, -32.50]
sec_lat5  = [-83.73, -68.96]
sec_lon6  = [LON[310,185], LON[370,185]] #185
sec_lat6  = [LAT[310,185], LAT[370,185]] #310 370
sec_lon7  = [LON[320,95], LON[450,95]] #95
sec_lat7  = [LAT[320,95], LAT[450,95]] #320 450
sec_lon8  = [LON[313,173], LON[380,173]] #173
sec_lat8  = [LAT[313,173], LAT[380,173]] #313 330
sec_lon9  = [LON[315,54], LON[415,54]] #54
sec_lat9  = [LAT[315,54], LAT[415,54]] #315 415
sec_lon10 = [LON[105,991], LON[180,991]] #991
sec_lat10 = [LAT[105,991], LAT[180,991]] #105 180
sec_lon11 = [LON[260,1148], LON[295,1148]] #1148
sec_lat11 = [LAT[260,1148], LAT[295,1148]] #260 295
sec_lon12 = [LON[255,1431], LON[400,1431]] #1431
sec_lat12 = [LAT[255,1431], LAT[400,1431]] #255 400



sec_i = [sec_lon1, sec_lon2, sec_lon3, sec_lon4, sec_lon5, sec_lon6, sec_lon7, sec_lon8, sec_lon9, sec_lon10, sec_lon11, sec_lon12]
sec_j = [sec_lat1, sec_lat2, sec_lat3, sec_lat4, sec_lat5, sec_lat6, sec_lat7, sec_lat8, sec_lat9, sec_lat10, sec_lat11, sec_lat12]

proj = ccrs.SouthPolarStereo() #ccrs.Mercator()
transform = ccrs.PlateCarree()
# ==============================================================================

# Loading domain geometry
ds_bat  = xr.open_dataset(bathy_file)
ds_cor  = xr.open_dataset(coord_file)

# Extracting only the part of the domain we need
ds_bat = ds_bat.isel(y=slice(0,600))
ds_cor = ds_cor.isel(y=slice(0,600))

# Plotting BATHYMETRY ----------------------------------------------------------

bathy = ds_bat["Bathymetry"].squeeze() #.isel(x_c=slice(1, None), y_c=slice(1, None))
env1 = ds_bat["hbatt_1"].squeeze()
env2 = ds_bat["hbatt_2"].squeeze()
env3 = ds_bat["hbatt_3"].squeeze()
lon = ds_cor["glamf"].squeeze()
lat = ds_cor["gphif"].squeeze()

for s in range(len(sec_i)):
    I = sec_i[s]
    J = sec_j[s]
    secI = []
    secJ = []
    for p in range(len(I)):
        j, i = get_ij_from_lon_lat(I[p], J[p], lon.data, lat.data)
        secI.append(i)
        secJ.append(j)
    SEC_J, SEC_I = get_poly_line_ij(np.asarray(secI), np.asarray(secJ))
    DEP = []
    EN1 = []
    EN2 = []
    EN3 = []
    LON = []
    LAT = []
    for z in range(len(SEC_I)):
        DEP.append(bathy.data[SEC_J[z], SEC_I[z]])
        EN1.append(env1.data[SEC_J[z], SEC_I[z]])
        EN2.append(env2.data[SEC_J[z], SEC_I[z]])
        EN3.append(env3.data[SEC_J[z], SEC_I[z]])
        LON.append(lon.data[SEC_J[z], SEC_I[z]])
        LAT.append(lat.data[SEC_J[z], SEC_I[z]])
    DEP = np.asarray(DEP)
    EN1 = np.asarray(EN1)
    EN2 = np.asarray(EN2)
    EN3 = np.asarray(EN3)
    LON = np.asarray(LON)
    LAT = np.asarray(LAT)

    fig = plt.figure(figsize = (34.,17.5), dpi=100)
    ax = fig.gca()
    zd = plt.plot(DEP, color='black')
    plt.setp(zd, 'linewidth', 4., zorder=20)
    e1 = plt.plot(EN1, '--', color='red')
    plt.setp(e1, 'linewidth', 2., zorder=30)
    e2 = plt.plot(EN2, '--', color='magenta')
    plt.setp(e2, 'linewidth', 2., zorder=30)
    e3 = plt.plot(EN3, '--', color='gold')
    plt.setp(e3, 'linewidth', 2., zorder=30)

    ax.set_ylim([0., maxdep])
    ax.invert_yaxis()    

    #ax.set_xlabel(xlabel, fontsize='50')
    ax.set_ylabel('Depth [m]', fontsize='50')
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)

    pos1 = ax.get_position()
    lon0 = -140.
    lon1 =  40.
    lat0 = -90.
    lat1 = -50.
 
    map_lims = [lon0, lon1, lat0, lat1]
    a = plt.axes([pos1.x0-0.002, pos1.y0+0.01, .3, .3], projection=proj)
    a.coastlines()
    a.add_feature(feature.LAND, color='gray',edgecolor='gray',zorder=1)
    MAP = plt.plot(LON, LAT, c="red", transform=transform)
    plt.setp(MAP, 'linewidth', 2.5)
    a.set_extent(map_lims)

    fig_name = 'sec_lon_'+str(LON[0])+'_lat_'+str(LAT[0])+'_'+str(maxdep)+'.png'
    fig_path = './'
    print(f"Saving {fig_path}", end=": ")
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    print("done")
    plt.close()
 
