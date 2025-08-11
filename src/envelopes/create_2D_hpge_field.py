#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | This module creates a 2D field of maximum spurious current |
#     | in the vertical and in time after an HPGE test.            |
#     | The resulting file can be used then to optimise the rmax   |
#     | of Multi-Envelope vertical grids.                          |
#     |                                                            |
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import os
from os.path import join, isfile, basename, splitext
import glob
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
#import xarray.ufuncs as xu
from dask.diagnostics import ProgressBar

# ==============================================================================
# Input files
# ==============================================================================

# Folder path containing HPGE spurious currents velocity files 
#HPGEdir = '/scratch/dbruciaf/GOSI10p1.0-hpge/r12_r16-r075-r040-r035_it2-r030'
HPGEdir = '/data/users/catherine.guiavarch/GOSI10/SouthernOcean/HPG_SO/u-dr701/run4_prof11/'

# List of indexes of the last T-level of each vertical subdomains 
# (Fortran indexening convention)
#num_lev = [74]
num_lev = [8,44,65,74]

# Name of the zonal and meridional velocity variables
Uvar = 'uo'
Vvar = 'vo'
# Name of the variable to chunk with dask and size of chunks
chunk_var = 'time_counter'
chunk_size = 1

# ==============================================================================
# LOOP

Ufiles = sorted(glob.glob(HPGEdir+'/*grid_U*.nc'))
Vfiles = sorted(glob.glob(HPGEdir+'/*grid_V*.nc'))

for F in range(len(Ufiles)):

    print(Ufiles[F])

    ds_U = xr.open_dataset(Ufiles[F], chunks={chunk_var:chunk_size})
    U4   = ds_U[Uvar]
    ds_V = xr.open_dataset(Vfiles[F], chunks={chunk_var:chunk_size})
    V4   = ds_V[Vvar]

    # rename some dimensions
    U4 = U4.rename({U4.dims[0]: 't', U4.dims[1]: 'k'})
    V4 = V4.rename({V4.dims[0]: 't', V4.dims[1]: 'k'})

    #print("U4 max:", np.nanmax(U4))
    #print("V4 max:", np.nanmax(V4))

    # interpolating from U,V to T
    U = U4.rolling({'x':2}).mean().fillna(0.)
    V = V4.rolling({'y':2}).mean().fillna(0.) 
    
    hpge = np.sqrt(np.power(U,2) + np.power(V,2))

    #print("hpge max:", np.nanmax(hpge))

    if F == 0:
       ni = hpge.data.shape[3]
       nj = hpge.data.shape[2]
       if len(num_lev) > 1:
          max_hpge1 = np.zeros(shape=(nj,ni))
          max_hpge2 = np.zeros(shape=(nj,ni))
          max_hpge3 = np.zeros(shape=(nj,ni))
          max_hpge4 = np.zeros(shape=(nj,ni))
       else:
          max_hpge1 = np.zeros(shape=(nj,ni))

    if len(num_lev) > 1:
       maxhpge_1 = hpge.isel(k=slice(None, num_lev[0])).max(dim='k').max(dim='t')
       maxhpge_2 = hpge.isel(k=slice(num_lev[0], num_lev[1])).max(dim='k').max(dim='t')
       maxhpge_3 = hpge.isel(k=slice(num_lev[1], num_lev[2])).max(dim='k').max(dim='t')
       maxhpge_4 = hpge.isel(k=slice(num_lev[2], num_lev[3])).max(dim='k').max(dim='t')
       max_hpge1 = np.maximum(max_hpge1, maxhpge_1.data)
       max_hpge2 = np.maximum(max_hpge2, maxhpge_2.data)
       max_hpge3 = np.maximum(max_hpge3, maxhpge_3.data)
       max_hpge4 = np.maximum(max_hpge4, maxhpge_4.data)
    else:
       maxhpge_1 = hpge.isel(k=slice(None, num_lev[0])).max(dim='k').max(dim='t')
       max_hpge1 = np.maximum(max_hpge1, maxhpge_1.data)

# Saving 
ds_hpge = xr.Dataset()
if len(num_lev) > 1:
   ds_hpge["max_hpge_1"] = xr.DataArray(max_hpge1, dims=('y','x'))
   ds_hpge["max_hpge_2"] = xr.DataArray(max_hpge2, dims=('y','x'))
   ds_hpge["max_hpge_3"] = xr.DataArray(max_hpge3, dims=('y','x'))
   ds_hpge["max_hpge_4"] = xr.DataArray(max_hpge4, dims=('y','x'))
else:
   ds_hpge["max_hpge_1"] = xr.DataArray(max_hpge1, dims=('y','x'))

# -------------------------------------------------------------------------------------   
# Writing the max_hpge file

print('WRITING the maximum_hpge.nc FILE')

out_file = "maximum_hpge.nc"
delayed_obj = ds_hpge.to_netcdf(join(HPGEdir,out_file), compute=False)

with ProgressBar():
     results = delayed_obj.compute()
