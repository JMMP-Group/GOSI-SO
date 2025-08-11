#!/usr/bin/env

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.feature as feature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import matplotlib.path as mpath

# ==============================================================================
# Input parameters

# 1. INPUT FILES

bat_file = "/data/users/frcg/GOSI10/SouthernOcean/GOSI-SO/create_plot/bathymetry.loc_area-antarctica.dep3550_sig3_stn9_itr3.nc"
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
bat = ds_bat.s2z_msk

# Get rid of discontinuity on lon grid 
#(from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(bat.coords["nav_lon"].diff("x", label="upper") > 0).cumprod("x").astype(bool)
bat.coords["nav_lon"] = (bat.coords["nav_lon"] + 360 * after_discont)
bat = bat.isel(x=slice(1, -1), y=slice(None, -1))

bat = bat.isel({'y':slice(0,800)})



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
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 50, 'color': 'k'}
gl.ylabel_style = {'size': 50, 'color': 'k'}
ax.set_extent([-180, 180, -90, -48], ccrs.PlateCarree())
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines

#cn_lev = [500., 1000., 2000., 3000., 4000., 4500., 5000.]
cn_lev = [0., 1., 2.]
#ax.contour(bat.nav_lon, bat.nav_lat, bat, levels=cn_lev,colors='black',linewidths=1.5, transform=transform, zorder=4)
#ax.contour(bat.nav_lon, bat.nav_lat, bat, levels=[5000],colors='forestgreen',linewidths=3., transform=transform, zorder=4)
#ax.contour(bat.nav_lon, bat.nav_lat, bat, levels=[5500],colors='blue',linewidths=3., transform=transform, zorder=4)
#ax.contour(bat.nav_lon, bat.nav_lat, bat, levels=[4000],colors='green',linewidths=3., transform=transform, zorder=4)
#ax.contourf(bat.nav_lon, bat.nav_lat, bat, levels=[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0], transform=transform, zorder=4,cmap='Accent')
ax.contourf(bat.nav_lon, bat.nav_lat, bat, levels=[-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0], transform=transform, zorder=4,cmap='Paired')
#ax.contourf(bat.nav_lon, bat.nav_lat, bat, levels=[2.0],colors='dodgerblue',linewidths=3., transform=transform, zorder=4)
#ax.contour(bat.nav_lon, bat.nav_lat, bat_gauss, levels=[3400],colors='magenta',linewidths=3., transform=transform, zorder=4)
#ax.contour(bat.nav_lon, bat.nav_lat, bat_gauss, levels=[3500],colors='deepskyblue',linewidths=3., transform=transform, zorder=4)
#ax.contour(bat.nav_lon, bat.nav_lat, bat_gauss, levels=[4350],colors='deepskyblue',linewidths=3., transform=transform, zorder=4)
#ax.contour(bat.nav_lon, bat.nav_lat, bat.nav_lat, levels=[-55.3],colors='red',linewidths=3., transform=transform, zorder=4)
#ax.contour(bat.nav_lon, bat.nav_lat, bat.nav_lat, levels=[-61.3],colors='blue',linewidths=3., transform=transform, zorder=4)


out_name ='bathy.png'
plt.savefig(out_name,bbox_inches="tight", pad_inches=0.1)
plt.close()

