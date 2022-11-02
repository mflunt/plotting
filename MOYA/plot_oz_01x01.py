#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 09:13:27 2020

Plot TOPOMI enhancements at 0.1x0.1 degrees. 

Use latitudinal mean in each 1 degree to subtract the background.

Want to look at winter 2019-2020 compared to 2018-2019.

@author: mlunt
"""

import xarray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colorbar as cbar
import glob
import pandas as pd
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def open_ds(fname):
    with xarray.open_dataset(fname) as ds:
        ds.load()
    return ds

def filenames(file_dir, file_str, file_type="nc", start=None, end=None, freq="D"):
    """
    Output a list of available file names,
    for given directory and date range.
    Assumes monthly files
    """
    files = []
    # Convert into time format
    if (start is not None) and (end is not None):
        days = pd.date_range(start = start, end = end, freq = freq)  #.to_pydatetime()
        if freq == "D":
            yearmonthday = [str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2) for d in days]
        elif freq == "MS":
            yearmonthday = [str(d.year) + str(d.month).zfill(2) for d in days]
    
        for ymd in yearmonthday:
            f=glob.glob(file_dir + "/" + file_str + ymd + "*." +file_type)
            if len(f) > 0:
                files += f
        files.sort()
        
    else:
        f=glob.glob(file_dir + "/" + file_str + "*." +file_type)
        if len(f) > 0:
            files += f     # Not entirely sure if this will work - might be worth checking! - Yes it works
        files.sort()
        ymd=""

    if len(files) == 0:
        print("Can't find file: " + file_dir + "/" + file_str + ymd[:4] + "*." + file_type)
                        
    return files

def read_netcdfs(files, dim = "time"):
    '''
    Use xray to open sequential netCDF or bpch files. 
    Makes sure that file is closed after open_dataset call.
    '''
    
    datasets = [open_ds(p) for p in sorted(files)]
    combined = xarray.concat(datasets, dim)
    
    return combined 

start = "20190101"
end = "20191231"

domain = "OZ"
#run_str = "SSA_BC_025x03125_CH4" 
output_dir  = "/geos/d21/mlunt/TROPOMI/processed/monthly_mean/" + domain + "/"
  

file_str = "TROPOMI_XCH4_01x01_" + domain + "_"
#file_str = "XCH4_Model_scaled_SRON_TROPOMI_"

# Too much memory for kingie - use holmes
files = filenames(output_dir, file_str, start = start, end=end, freq="MS")

ds_temp = read_netcdfs(files)

#%%
zonal_mean = ds_temp.XCH4.mean(dim="lon")

# Fit a function to this zonal mean - straight line

# Subtract this straight line function to form enhancements

ntime = len(ds_temp.time)
lon=ds_temp.lon.values
lat = ds_temp.lat.values
nlat = len(lat)
ch4_bl=zonal_mean.copy() *0.

for ti in range(ntime):

    idx = np.isfinite(zonal_mean[ti,:])
    pfit = np.poly1d(np.polyfit(lat[idx],zonal_mean[ti,idx], 1))
    
    ch4_bl[ti,:] = pfit(lat)
    
#%%
#reg_lats = [-44,-10.]
#reg_lons =  [100.,156.]  

#reg_lats = [-25,-20.]
#reg_lons =  [145.,150.]

#reg_lats = [-30,-20.]
#reg_lons =  [145.,155.]   

# Hunter valley
#reg_lats = [-34,-28.]
#reg_lons =  [148.,155.]   

#reg_lats = [40.,60.]
#reg_lons =  [-12.,15.]   
    
# WA 
reg_lats = [-44,-10.]
reg_lons =  [100.,156.]  
    
ds_reg = ds_temp.sel(lat=slice(reg_lats[0],reg_lats[1]), lon=slice(reg_lons[0],reg_lons[1]))    

xch4 = ds_reg.XCH4    
ch4_plus2 = ds_reg.XCH4 - ch4_bl

albedo = ds_reg.SWIR_albedo
aod = ds_reg.SWIR_AOD

ch4_plus = ch4_plus2.where((aod<0.1) & (albedo < 0.3) & (albedo >= 0.05))

lat_reg=  ds_reg.lat.values
lon_reg = ds_reg.lon.values

proj = ccrs.PlateCarree()


#%%
fig,axes=plt.subplots(2,2,figsize=(10,8), subplot_kw=dict(projection=proj),
                      sharex=True, sharey=True)
axs=axes.ravel()
base_month=0

for xi in range(4):
    p0 = axs[xi].pcolormesh(lon_reg, lat_reg, 
                    ch4_plus[xi+base_month,:,:],
                    transform=ccrs.PlateCarree(), cmap='Spectral_r', vmin=0.,vmax=20.)
    axs[xi].coastlines()
    axs[xi].add_feature(cfeature.BORDERS, edgecolor='grey', alpha=0.7)

#for xi in range(16):
#    p0 = axs[xi].pcolormesh(lon, lat, 
#                ch4_plus[xi+base_month,:,:],
#                transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-5.,vmax=50.)
#    #p0 = axs[xi].pcolormesh(lon, lat, 
#    #            x_post[xi+base_month,:,:],
#    #            transform=ccrs.PlateCarree(), cmap='PuRd', vmin=0.,vmax=6.)
#    axs[xi].coastlines()
#    axs[xi].add_feature(cfeature.BORDERS, edgecolor='grey', alpha=0.7)
    
    ti_temp = ch4_plus.time[xi+base_month]
    axs[xi].set_title(str(ti_temp.dt.year.values) + "-" + str(ti_temp.dt.month.values))
    
cbaxes = fig.add_axes([0.1, 0.08, 0.8, 0.02]) 
##[left, bottom, width, height],
#cb = plt.colorbar(p0, cax = cbaxes, orientation='horizontal', extend='both', label = 'Posterior scale factor') 
cb = plt.colorbar(p0, cax = cbaxes, orientation='horizontal', extend='both', label = 'Monthly mean XCH4 enhancement (ppb)')

#%%
#data=ch4_plus.mean(dim="time")

rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',
                                        facecolor='none')
                                        

#data=xch4.mean(dim="time")
data=aod.mean(dim="time")

fig2,ax2=plt.subplots(figsize=(10,8), subplot_kw=dict(projection=proj))

p2 = ax2.pcolormesh(lon_reg, lat_reg, 
                    data,
                    transform=ccrs.PlateCarree(), cmap='Spectral_r', vmin=0.,vmax=0.3)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, edgecolor='grey', alpha=0.7)
#ax2.add_feature(cfeature.RIVERS, edgecolor='blue')
ax2.add_feature(rivers,edgecolor='blue', facecolor='none')
cbaxes2 = fig2.add_axes([0.1, 0.08, 0.8, 0.02]) 
##[left, bottom, width, height],
#cb = plt.colorbar(p0, cax = cbaxes, orientation='horizontal', extend='both', label = 'Posterior scale factor') 
cb2 = plt.colorbar(p2, cax = cbaxes2, orientation='horizontal', extend='both', label = 'Annual mean XCH4 (ppb)')

#%%
data2=ch4_plus.mean(dim="time")

fig2,ax2=plt.subplots(figsize=(10,8), subplot_kw=dict(projection=proj))

p2 = ax2.pcolormesh(lon_reg, lat_reg, 
                    data2,
                    transform=ccrs.PlateCarree(), cmap='Spectral_r', vmin=0.,vmax=20.)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, edgecolor='grey', alpha=0.7)
cbaxes2 = fig2.add_axes([0.1, 0.08, 0.8, 0.02]) 
##[left, bottom, width, height],
#cb = plt.colorbar(p0, cax = cbaxes, orientation='horizontal', extend='both', label = 'Posterior scale factor') 
cb2 = plt.colorbar(p2, cax = cbaxes2, orientation='horizontal', extend='both', label = 'Monthly mean XCH4 enhancement (ppb)')

gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')

gl.xlocator = mticker.FixedLocator([110,120,130,135,136,137,138,139,140, 147, 148,149,150,151,152,153, 154])
gl.ylocator = mticker.FixedLocator([-40,--35,-34,-33,-32,30,-28,-26,-24,-22,-20,-19,-18,-17,-16,-12,-8])#            "20180701", "20180716",
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER