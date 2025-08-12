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
from lib import compute_masks

# ==============================================================================
# Input files
# ==============================================================================

# File describing the geometry of the model domain (i.e., domain_cfg.nc)
DOMfile = '/data/users/catherine.guiavarch/GOSI10/SouthernOcean/GOSI-SO/GOSI-SO_inputs/domain_cfg_antarctica.dep3550_sig3_stn9_itr3_r12_r12.nc'

# File including the localisation mask
LOCfile = '/data/users/catherine.guiavarch/GOSI10/SouthernOcean/GOSI-SO/GOSI-SO_inputs/bathymetry.loc_area-antarctica.dep3550_sig3_stn9_itr3.MEs_4env_3550_r12_r12.nc'

# Folder path containing HPGE spurious currents velocity files 
HPGEdir = '/data/scratch/diego.bruciaferri/HPG_SO/u-dr701/run3_prof10/'

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

# Computing volume of model cells
ds_d = xr.open_dataset(DOMfile, chunks={}, drop_variables=("nav_lev","time_counter"))
ds_d = ds_d.rename_dims({'nav_lev':'z','time_counter':'t'}).squeeze()
ds_d = compute_masks(ds_d, merge=True)
volT = ds_d.e1t * ds_d.e2t * ds_d.e3t_0
volU = ds_d.e1u * ds_d.e2u * ds_d.e3u_0
volV = ds_d.e1v * ds_d.e2v * ds_d.e3v_0

# Localisation mask
ds_l = xr.open_dataset(LOCfile)
Lmsk = ds_l.s2z_msk
Lmsk = Lmsk.where(Lmsk==0,1)

# LOOP

Ufiles = sorted(glob.glob(HPGEdir+'/*grid_U*.nc'))
Vfiles = sorted(glob.glob(HPGEdir+'/*grid_V*.nc'))

for F in range(len(Ufiles)):

    print(Ufiles[F])

    ds_U = xr.open_dataset(Ufiles[F], chunks={chunk_var:chunk_size}, 
                                      drop_variables=("depthu","time_counter")
           )
    u4   = ds_U[Uvar]
    ds_V = xr.open_dataset(Vfiles[F], chunks={chunk_var:chunk_size}, 
                                      drop_variables=("depthv","time_counter")
           )
    v4   = ds_V[Vvar]

    # rename some dimensions
    u4 = u4.rename({"time_counter": 't', "depthu": 'z'})
    v4 = v4.rename({"time_counter": 't', "depthv": 'z'})

    #print(u4)
    #print(u4.t)
    #print(v4.shape)

    #print("U4 max:", np.nanmax(U4))
    #print("V4 max:", np.nanmax(V4))

    # interpolating from U,V to T
    UT = (u4 * volU).rolling({'x':2}).mean().fillna(0.) / volT
    VT = (v4 * volV).rolling({'y':2}).mean().fillna(0.) / volT 
    
    hpge = np.sqrt(np.power(UT,2) + np.power(VT,2)) * ds_d.tmask

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
       maxhpge_1 = hpge.isel(z=slice(None, num_lev[0])).max(dim='z').max(dim='t')
       maxhpge_2 = hpge.isel(z=slice(num_lev[0], num_lev[1])).max(dim='z').max(dim='t')
       maxhpge_3 = hpge.isel(z=slice(num_lev[1], num_lev[2])).max(dim='z').max(dim='t')
       maxhpge_4 = hpge.isel(z=slice(num_lev[2], num_lev[3])).max(dim='z').max(dim='t')
       max_hpge1 = np.maximum(max_hpge1, maxhpge_1.data)
       max_hpge2 = np.maximum(max_hpge2, maxhpge_2.data)
       max_hpge3 = np.maximum(max_hpge3, maxhpge_3.data)
       max_hpge4 = np.maximum(max_hpge4, maxhpge_4.data)
    else:
       maxhpge_1 = hpge.isel(z=slice(None, num_lev[0])).max(dim='z').max(dim='t')
       max_hpge1 = np.maximum(max_hpge1, maxhpge_1.data)

# Saving 
ds_hpge = xr.Dataset()
if len(num_lev) > 1:
   ds_hpge["max_hpge_1"] = xr.DataArray(max_hpge1*Lmsk, dims=('y','x'))
   ds_hpge["max_hpge_2"] = xr.DataArray(max_hpge2*Lmsk, dims=('y','x'))
   ds_hpge["max_hpge_3"] = xr.DataArray(max_hpge3*Lmsk, dims=('y','x'))
   ds_hpge["max_hpge_4"] = xr.DataArray(max_hpge4*Lmsk, dims=('y','x'))
else:
   ds_hpge["max_hpge_1"] = xr.DataArray(max_hpge1*Lmsk, dims=('y','x'))

# -------------------------------------------------------------------------------------   
# Writing the max_hpge file

print('WRITING the maximum_hpge.nc FILE')

out_file = "maximum_hpge.nc"
delayed_obj = ds_hpge.to_netcdf(join(HPGEdir,out_file), compute=False)

with ProgressBar():
     results = delayed_obj.compute()
