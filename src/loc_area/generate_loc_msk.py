#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | This module creates general envelope surfaces to be        |
#     | used to generate a Localised Multi-Envelope s-coordinate   |
#     | vertical grid.                                             |
#     |                                                            |
#     |                                                            |
#     |                      -o#&&*''''?d:>b\_                     |
#     |                 _o/"`''  '',, dMF9MMMMMHo_                 |
#     |              .o&#'        `"MbHMMMMMMMMMMMHo.              |
#     |            .o"" '         vodM*$&&HMMMMMMMMMM?.            |
#     |           ,'              $M&ood,~'`(&##MMMMMMH\           |
#     |          /               ,MMMMMMM#b?#bobMMMMHMMML          |
#     |         &              ?MMMMMMMMMMMMMMMMM7MMM$R*Hk         |
#     |        ?$.            :MMMMMMMMMMMMMMMMMMM/HMMM|`*L        |
#     |       |               |MMMMMMMMMMMMMMMMMMMMbMH'   T,       |
#     |       $H#:            `*MMMMMMMMMMMMMMMMMMMMb#}'  `?       |
#     |       ]MMH#             ""*""""*#MMMMMMMMMMMMM'    -       |
#     |       MMMMMb_                   |MMMMMMMMMMMP'     :       |
#     |       HMMMMMMMHo                 `MMMMMMMMMT       .       |
#     |       ?MMMMMMMMP                  9MMMMMMMM}       -       |
#     |       -?MMMMMMM                  |MMMMMMMMM?,d-    '       |
#     |        :|MMMMMM-                 `MMMMMMMT .M|.   :        |
#     |         .9MMM[                    &MMMMM*' `'    .         |
#     |          :9MMk                    `MMM#"        -          |
#     |            &M}                     `          .-           |
#     |             `&.                             .              |
#     |               `~,   .                     ./               |
#     |                   . _                  .-                  |
#     |                     '`--._,dd###pp=""'                     |
#     |                                                            |
#     |                                                            |
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 24-05-2022, Met Office, UK                 |
#     |------------------------------------------------------------|


import sys
import os
from os.path import isfile, basename, splitext
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from scipy.ndimage import gaussian_filter

from lib import *

# ==============================================================================
# 1. Checking for input files
# ==============================================================================

msg_info('LOCALISATION AREA GENERATOR', main=True)

# Checking input file for envelopes
if len(sys.argv) != 2:
   raise TypeError(
         'You need to specify the absolute path of the localisation input file'
   )

# Reading local area infos
loc_file = sys.argv[1]
msg_info('Reading localisation parameters ....')
locInfo = read_locInfo(loc_file)

bathyFile = locInfo.bathyFile
hgridFile = locInfo.hgridFile
loc_isos  = locInfo.loc_isos
loc_area  = locInfo.loc_area
s2z_factr = locInfo.s2z_factr
s2z_sigma = locInfo.s2z_sigma
s2z_stenc = locInfo.s2z_stenc
s2z_itera = locInfo.s2z_itera
s2z_wghts = locInfo.s2z_wghts

tol = 1.e-7

# Reading bathymetry and horizontal grid
msg_info('Reading bathymetry data ... ')
ds_bathy = xr.open_dataset(bathyFile).squeeze()
ds_bathy = ds_bathy.fillna(0.)
msg_info('Reading horizontal grid data ... ')
ds_grid = xr.open_dataset(hgridFile)
glamt = ds_grid["glamt"].squeeze()
gphit = ds_grid["gphit"].squeeze()

if "nav_lon" in ds_bathy:
   ds_bathy = ds_bathy.set_coords(["nav_lon","nav_lat"])
else:
   ds_bathy["nav_lon"] = glamt
   ds_bathy["nav_lat"] = gphit
   ds_bathy = ds_bathy.set_coords(["nav_lon","nav_lat"])
bathy = ds_bathy["Bathymetry"] #.fillna(0.).squeeze()

# Defining local variables -----------------------------------------------------

ni      = glamt.shape[1]
nj      = glamt.shape[0]
ds_loc  = ds_bathy.copy() 

# Computing LSM
lsm = xr.where(bathy > 0, 1, 0) 

msg = 'Creating areas for localising a vertical coordinate system ... '
msg_info(msg)

# Create mask identifying zone where 
# we want local coordinates
msk_loc = generate_loc_area(bathy, loc_isos)
ds_loc["loc_area"] = msk_loc
msk_loc.plot()
plt.show()

# Creating the transition zone: gaussian filtering 
# for identifying limits of transition zone
msk_wrk = xr.where(msk_loc==1, 1., 1.+s2z_factr)
wrk = msk_wrk.copy()
s2z_trunc = (((s2z_stenc - 1.)/2.)-0.5)/s2z_sigma
for nit in range(s2z_itera):
    zwrk_gauss = gaussian_filter(wrk, sigma=s2z_sigma, truncate=s2z_trunc)
    zwrk_gauss = msk_wrk.where(msk_loc==1, zwrk_gauss)
    wrk = zwrk_gauss.copy()

# Mask identifying zones
#    global      = 0
#    loc to glob = 1
#    local       = 2
msk_zones = xr.full_like(msk_loc, None, dtype=np.double)
msk = np.ones(shape=msk_zones.values.shape)

diff = np.absolute(zwrk_gauss - msk_wrk)
values, counts = np.unique(diff, return_counts=True)
ind = np.argmax(counts)
print(values[ind])  # prints the most frequent element
diff = diff.where(diff != values[ind])
msk[np.isnan(diff)] = 0
msk[msk_loc==1] = 2
msk_zones.data = msk
ds_loc["s2z_msk"] = msk_zones

# 4. Weights to generate transitioning envelope
if s2z_wghts:
   msg = 'Computing weights ...'
   msg_info(msg)
   weights = weighting_dist(msk_zones, a=1.7)
   da_wgt = xr.full_like(bathy,None,dtype=np.double)
   da_wgt.data = weights
   ds_loc["s2z_wgt"] = da_wgt

# -------------------------------------------------------------------------------------   
# Writing the bathy_meter.nc file

msg = 'WRITING the bathy_meter.nc FILE'
msg_info(msg)

if loc_isos >= 0: 
   isos = "dep"
else:
   isos = "lat"
out_name = '.' + isos + str(int(loc_isos)) + '_sig' + str(s2z_sigma) + '_stn' + str(s2z_stenc) + '_itr' + str(s2z_itera)
out_file = "bathymetry.loc_area-" + loc_area + out_name + ".nc"
ds_loc.to_netcdf(out_file)
