#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 13-08-2024, Met Office, UK                 |
#     |------------------------------------------------------------|


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xarray as xr
from utils import calc_r0, prepare_domcfg, plot_sec

# ==============================================================================
# Input parameters

fig_path = './'

# 1. INPUT FILES

vvar = None #"r0x" # None
fig_path = './'

# ----- GOSI10
domcfg = ['/data/users/dbruciaf/SO/domain_cfg.3550_r12_r12_v2.nc']#,
#          '/data/users/dbruciaf/GOSI10_input_files/p1.0/domcfg_eORCA025_v3.1_r42_cut_novf.nc']
fbathy = ['/data/users/dbruciaf/SO/bathymetry.loc_area-antarctica.dep3550_sig3_stn9_itr1.MEs_4env_3550_r12_r12.nc']#,
#          None]


# Define the section we want to plot:
list_sec = [
            {'lon':[154.81, 174.99, -174. ] ,
             'lat':[-80.39, -77.96, -67.30]},
            {'lon':[172.39, -162. ] ,
             'lat':[-84.19, -71.09]}, # i = 1010
            {'lon':[-106.13, -108.50] , 
             'lat':[-75.92 , -67.58]}, # i = 1011
            {'lon':[-83.89, -43. ] ,
             'lat':[-79.92, -67.96]}, # i = 1012
            {'lon':[-64.30, -32.50] ,
             'lat':[-83.73, -68.96]}, # j = 1058
            #{'lon':[119.25, 119.25],
            # 'lat':[-67.86, -61.49]},
            #{'lon':[96.75, 96.75],
            # 'lat':[-66.91, -50.38]},
            #{'lon':[116.25, 116.25],
            # 'lat':[-67.582417, -65.908232]},
            #{'lon':[86.5, 86.5],
            # 'lat':[-67.393977, -55.64]},
            #{'lon':[-38.532573, -39.136015],
            # 'lat':[-81.312509, -77.456065]},
            #{'lon':[0.026757, 0.000748],
            # 'lat':[-72.056199, -69.225782]},
            #{'lon':[70.75, 70.75],
            # 'lat':[-72.508768, -57.70]},
           ]

# ==============================================================================

for exp in range(len(domcfg)):

    # Loading domain geometry
    ds_dom, hbatt, vcoor = prepare_domcfg(domcfg[exp], fbathy[exp])

    hbatt = [] # TODO: use realistic envelopes

    # Computing slope paramter of model levels if needed
    r0_3D = ds_dom.gdept_0 * 0.0
    if vvar is not None:
       r0_3D = r0_3D.rename(vvar)
       for k in range(r0_3D.shape[0]):
           r0 = calc_r0(ds_dom.gdept_0.isel(z=k))
           r0_3D[k,:,:] = r0
    else:
       vvar = 'dummy'
    r0_3D = r0_3D.where(ds_dom.tmask > 0)
    ds_dom[vvar] = r0_3D

    # Extracting variables for the specific type of section
    var_list = ["gdepu_0" , "gdepuw_0", "gdepv_0" , "gdepvw_0",
                "gdept_0" , "gdepw_0" , "gdepf_0" , "gdepfw_0",
                "glamt"   , "glamu"   , "glamv"   , "glamf"   ,
                "gphit"   , "gphiu"   , "gphiv"   , "gphif"   ,
                "gdepw_1d", "loc_msk" , vvar]

    for coord_sec in list_sec:

        sec_lon = coord_sec['lon']
        sec_lat = coord_sec['lat']

        print ('section through lon:', sec_lon)
        print ('                lat:', sec_lat)

        ds_sec = ds_dom[var_list]
        ds_var = ds_dom[var_list]

        sec_name = str(sec_lon[0])+'-'+str(sec_lat[0])+'_'+str(sec_lon[-1])+'-'+str(sec_lat[-1])

        if vvar == "dummy":
           fig_name = vcoor+'_section_'+sec_name+'.png'
        else:
           fig_name = vcoor+'_section_'+vvar+'_'+sec_name+'.png'
        plot_sec(fig_name, fig_path, ds_sec, ds_var, vvar, sec_lon, sec_lat, hbatt, imap=True)

