#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:41:39 2022

Plot 2020-2021 GOSAt anomaly

or effectively plot growth rate anomaly

@author: mlunt
"""

import xarray
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colorbar as cbar
from scipy import stats
import areagrid

def open_ds(fname, group=None):
    with xarray.open_dataset(fname, group) as ds:
        ds.load()
    return ds

def filenames(start, end, file_dir, file_string, freq="MS"):
    """
    Output a list of available file names,
    for given directory and date range.
    Assumes monthly files
    """
    
    # Convert into time format
    #days = pd.DatetimeIndex(start = start, end = end, freq = "1D").to_pydatetime()
    
    days = pd.date_range(start = start, end = end, freq = freq)
    
    
    if freq == "D":
        yrmnd = [str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2) for d in days]
    elif freq == "MS":
        yrmnd = [str(d.year) + str(d.month).zfill(2) for d in days]
    elif freq == "YS":
        yrmnd = [str(d.year) for d in days]
    
    #yrmnd = [str(d.year) + str(d.month).zfill(2) for d in days]

    files = []
    for ymd in yrmnd:
        f=glob.glob(file_dir  + "/" + file_string +  
                    ymd + "*.nc")
        if len(f) > 0:
            files += f
    files.sort()

    if len(files) == 0:
        print("Can't find file: " + file_dir + "/" + file_string + ymd +  "*.nc")
                        
    return files


def read_netcdfs(files, dim = "time"):
    '''
    Use xray to open sequential netCDF files. 
    Makes sure that file is closed after open_dataset call.
    '''
    datasets = [open_ds(p) for p in sorted(files)]
    combined = xarray.concat(datasets, dim)
    return combined   

start_date = "20100101"
end_date = "20211231"

data_dir = "/geos/d21/mlunt/GOSAT/processed/GLOBAL/"

f_str =  "GOSAT_XCH4_2x25_monthly_GLOBAL_" 
#f_str =  "GOSAT_XCH4_05x05_monthly_GLOBAL_" 

files = filenames(start_date, end_date, data_dir, f_str, freq="YS")
ds = read_netcdfs(files, dim="time")

xch4 = ds.XCH4
flag = ds.retr_flag

xch4_land  = xch4.where(flag == 0)

zonal_mn = xch4.quantile(0.1,dim="lon")
xch4_anom = xch4_land - zonal_mn

global_mn = xch4.mean(dim=["lon","lat"])
tropic_mn = zonal_mn.sel(lat=slice(0,20)).mean(dim="lat")

#xch4_2020 = xch4.sel(time=slice("20200101", "20211230")).mean(dim="time")
#xch4_2020 = xch4.sel(time=slice("20200101", "20201230")).mean(dim="time")
#xch4_2019 = xch4.sel(time=slice("20190101", "20191230")).mean(dim="time")

xch4_2021 = xch4.sel(time=slice("20210101", "20211230")).mean(dim="time")
xch4_2020 = xch4.sel(time=slice("20200101", "20201230")).mean(dim="time")
xch4_2019 = xch4.sel(time=slice("20190101", "20191230")).mean(dim="time")


diff_2019 = xch4_2020  - xch4_2019
diff_zonal_mn_2019 = diff_2019.mean(dim="lon")
diff_global_mn_2019 = diff_2019.mean()

diff_2120 = xch4_2021  - xch4_2020
diff_zonal_mn_2120 = diff_2120.mean(dim="lon")
diff_global_mn_2120 = diff_2120.mean()

#xch4_land_2020 = xch4_land.sel(time=slice("20200101", "20201230")).mean(dim="time")
#xch4_land_2019 = xch4_land.sel(time=slice("20190101", "20191230")).mean(dim="time")

xch4_land_2021 = xch4_land.sel(time=slice("20210101", "20211230")).mean(dim="time")
xch4_land_2020 = xch4_land.sel(time=slice("20200101", "20201230")).mean(dim="time")
xch4_land_2019 = xch4_land.sel(time=slice("20190101", "20191230")).mean(dim="time")

diff_land_2019 = xch4_land_2020  - xch4_land_2019

#diff_zonal_mn2 = diff_zonal_mn.reindex_like(dum_zonal_mn, method="nearest")
diff2_2019 = diff_2019 - diff_zonal_mn_2019
diff2_2120 = diff_2120 - diff_zonal_mn_2120

diff_land2 = diff_land_2019 - diff_zonal_mn_2019

lon = diff_2019.lon.values
lat = diff_2019.lat.values

states = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale='50m',
            facecolor='none')

rivers10 = cfeature.NaturalEarthFeature(scale='10m', category='physical',
        name='rivers_lake_centerlines', facecolor='none')

rivers50 = cfeature.NaturalEarthFeature(scale='50m', category='physical',
        name='rivers_lake_centerlines', facecolor='none')


#%%

xch4_1819 = xch4_land.sel(time=slice("20190101", "20191230")).mean(dim="time")
xch4_2021 = xch4_land.sel(time=slice("20200101", "20211231")).mean(dim="time")

diff_2yr = xch4_2021 - xch4_1819
diff_2yr_mn = diff_2yr.mean()
diff_2yr_zonal  =diff_2yr.mean(dim="lon")
#diff_2yr_anom = diff_2yr - diff_2yr_zonal
diff_2yr_anom = diff_2yr - diff_2yr_mn

cmin=-12
cmax = 12

#proj = ccrs.PlateCarree()
proj = ccrs.Robinson()
fig3,ax3=plt.subplots(subplot_kw=dict(projection=proj),figsize=(10,8))

#ax3.add_feature(states, edgecolor='grey', alpha=0.5)
h3 = ax3.pcolormesh(lon, lat, diff_2yr_anom, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=cmin, vmax=cmax)
#h3 = ax3.pcolormesh(lon, lat, diff2, transform=ccrs.PlateCarree(), cmap='RdPu', vmin=cmin, vmax=cmax)
#h3 = ax3.pcolormesh(lon_out, lat_out, grid_mean, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=1780, vmax=1880)
#ax3.set_extent([20,50,-8,20])
#ax3.add_feature(states, edgecolor='grey', alpha=0.5)
cax3,kw3 = cbar.make_axes(ax3,location='bottom',pad=0.02,shrink=0.9)
cb=fig3.colorbar(h3,cax=cax3,extend='both',**kw3)
ax3.coastlines()
#ax3.add_feature(rivers10, edgecolor='dodgerblue')

cb.set_label('XCH$_4$ growth rate anomaly (ppb)', size=12)

#%%

diff_monthly = xch4.sel(time=slice("20210101", "20211230")) - xch4.sel(time=slice("20190101", "20191230")).values
diff_land_monthly = xch4_land.sel(time=slice("20210101", "20211230")) - xch4_land.sel(time=slice("20190101", "20191230")).values

diff_month_zonal = diff_monthly.mean(dim="lon")
diff_month_land = diff_land_monthly - diff_month_zonal

fig,axes = plt.subplots(4,3, subplot_kw=dict(projection=proj),figsize=(8,8), 
                        sharex=True, sharey=True)
axs=axes.ravel()
cmin2 = -20
cmax2=20

for ti in range(12):
    
    h3 = axs[ti].pcolormesh(lon, lat, diff_month_land[ti,:,:], transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=cmin2, vmax=cmax2)
    axs[ti].coastlines()
    
    #axs[ti].set_extent([100,180,-40,-10])

