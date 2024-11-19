#!/usr/bin/env

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.feature as feature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================================
# Input parameters

# 1. INPUT FILES

bat_file = "/data/users/dbruciaf/SO/bathy_eORCA025_Storkey2024_v1.3_isf_handedit.nc"
cor_file = "/data/users/dbruciaf/GOSI10_input_files/p1.0/domcfg_eORCA025_v3.1_r42.nc" 

# 2. PLOT
proj = ccrs.SouthPolarStereo() #ccrs.Robinson()

# ==============================================================================

# Load localisation file

ds_bat = xr.open_dataset(bat_file)
ds_cor = xr.open_dataset(cor_file)

# Taking care of longitude discontinuity
ds_bat.coords["x"] = range(ds_bat.dims["x"])
ds_bat.coords["y"] = range(ds_bat.dims["y"])
ds_bat = ds_bat.assign_coords({"nav_lon":ds_cor.nav_lon, "nav_lat":ds_cor.nav_lat})

# Extracting variables
bat = ds_bat.Bathymetry

# Get rid of discontinuity on lon grid 
#(from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(bat.coords["nav_lon"].diff("x", label="upper") > 0).cumprod("x").astype(bool)
bat.coords["nav_lon"] = (bat.coords["nav_lon"] + 360 * after_discont)
bat = bat.isel(x=slice(1, -1), y=slice(None, -1))

bat = bat.isel({'y':slice(0,600)})

fig = plt.figure(figsize=(25, 25), dpi=100)
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
ax = fig.add_subplot(spec[:1], projection=proj)
ax.coastlines(linewidth=4, zorder=6)
ax.add_feature(feature.LAND, color='gray',edgecolor='black',zorder=5)

# Drawing settings
transform = ccrs.PlateCarree()

# Grid settings
gl_kwargs = dict()
gl = ax.gridlines(**gl_kwargs)
gl.xlines = False
gl.ylines = False
gl.top_labels = True
gl.right_labels = True
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 50, 'color': 'k'}
gl.ylabel_style = {'size': 50, 'color': 'k'}

cn_lev = [500., 1000., 2000., 3000., 4000., 4500., 5000.]
ax.contour(bat.nav_lon, bat.nav_lat, bat, levels=cn_lev,colors='black',linewidths=1.5, transform=transform, zorder=4)
ax.contour(bat.nav_lon, bat.nav_lat, bat, levels=[5000],colors='forestgreen',linewidths=3., transform=transform, zorder=4)
ax.contour(bat.nav_lon, bat.nav_lat, bat, levels=[5500],colors='blue',linewidths=3., transform=transform, zorder=4)

ax.contour(bat.nav_lon, bat.nav_lat, bat.nav_lat, levels=[-56.3],colors='red',linewidths=6., transform=transform, zorder=4)

out_name ='bathy.png'
plt.savefig(out_name,bbox_inches="tight", pad_inches=0.1)
plt.close()

