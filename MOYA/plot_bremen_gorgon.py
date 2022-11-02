#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:41:50 2022

Plot Gorgon/barrow Island enhancement

@author: mlunt
"""
import xarray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#from areagrid import areagrid
import glob

import matplotlib.colorbar as cbar
import re
from cartopy.io.img_tiles import GoogleTiles as OSM


def open_ds(fname, group=None):
    with xarray.open_dataset(fname, group) as ds:
        ds.load()
    return ds

def filenames(start, end, file_dir, file_string):
    """
    Output a list of available file names,
    for given directory and date range.
    Assumes monthly files
    """
    
    # Convert into time format
    #days = pd.DatetimeIndex(start = start, end = end, freq = "1D").to_pydatetime()
    
    days = pd.date_range(start = start, end = end, freq = "1D")
    yrmnd = [str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2) for d in days]

    files = []
    for ymd in yrmnd:
        f=glob.glob(file_dir  + "/" + file_string +  
                    ymd + "*.nc")
        if len(f) > 0:
            files += f
    files.sort()

    if len(files) == 0:
        print("Can't find file: " + file_dir + "/" + file_str + ymd +  "*.nc")
                        
    return files

def read_netcdfs(files, dim = "time"):
    '''
    Use xray to open sequential netCDF files. 
    Makes sure that file is closed after open_dataset call.
    '''
    datasets = [open_ds(p) for p in sorted(files)]
    combined = xarray.concat(datasets, dim)
    return combined   

#%%
year = "2019"   # 2020 06
month = "11"

domain = "Gorgon"

qa_threshold = 0.5


# N. America
if domain == "Gorgon":
    lonmin = 115.
    lonmax = 116.
    latmin = -22.
    latmax= -20.5


lat_range = latmax-latmin
lat_mid = (latmin+latmax)/2.

# SSA

dlat=0.1
dlon=0.1

lon_out = np.arange(lonmin,lonmax+dlon, dlon)
lat_out = np.arange(latmin,latmax+dlat, dlat)


nlat = len(lat_out)
nlon = len(lon_out)
nlev = 20
lev_out = np.arange(nlev)

data_dir = "/home/mlunt/ceph/measurements/satellite/TROPOMI/CH4/bremen/" + year + "/"
file_str = "ESACCI-GHG-L2-CH4-CO-TROPOMI-WFMD-"  #20200101-fv2.nc

start_date = year + month + "01"

if month in(["04", "06", "09", "11"]):
    end_date = year + month + "30"
elif month in (["02"]):
    end_date = year + month + "28"
else:
    end_date = year + month + "31"

#start_date = year+month+"01"
#end_date = year+month+"31"

files = filenames(start_date, end_date, data_dir, file_str)
nfiles = len(files)

#%%   
xch4_list=[]
sat_lat_list=[]
sat_lon_list=[]
date_list=[]
for ti, file in enumerate(files):
    
    ds = open_ds(file)
    #file_date = pd.to_datetime(ds.time[0].values)
    file_date = pd.to_datetime(re.findall("([0-9]{8})", file)[-1])
    fdate_str = re.findall("([0-9]{8})", file)[-1]
    #dates.append(file_date)
 
    ds_quality2 = ds.where(ds.xch4_quality_flag == 0, drop=True)
    
    # Filter out high and low lats as they're just not needed:
    ds_quality = ds_quality2.where((ds_quality2.latitude >= latmin) &
                                   (ds_quality2.latitude < latmax) &
                                   (ds_quality2.longitude >= lonmin) &
                                   (ds_quality2.longitude < lonmax), 
                                   drop=True)
    
    
    xch4 = ds_quality.xch4.values
    if len(xch4) > 0:
    
        sat_lat = ds_quality.latitude.values
        sat_lon = ds_quality.longitude.values
        
        date_list.append(file_date)
        xch4_list.append(xch4)
        sat_lat_list.append(sat_lat)
        sat_lon_list.append(sat_lon)
        
#%%
tiler = OSM(style='satellite')
mercator = tiler.crs
map_extents = [lonmin, lonmax, latmin, latmax]
proj = ccrs.PlateCarree()
ndates = len(date_list[:10])
for ti in range(5,5+ndates):
    
    fig,ax = plt.subplots(subplot_kw=dict(projection=proj))
    
    ax.set_extent(map_extents)
    ax.add_image(tiler,10)

    
    p2 = ax.scatter(sat_lon_list[ti], sat_lat_list[ti], 
                        c=xch4_list[ti],
                        transform=ccrs.PlateCarree(), cmap='Spectral_r', vmin=1800.,vmax=1840.)
    ax.set_title(date_list[ti].strftime('%Y%m%d'))
    
    #cbaxes2 = fig.add_axes([0.1, 0.08, 0.8, 0.02]) 
    cbaxes2 = fig.add_axes([0.8, 0.2, 0.03, 0.6]) 
    ##[left, bottom, width, height],

    cb2 = plt.colorbar(p2, cax = cbaxes2, orientation='vertical', extend='both', label = 'XCH4  (ppb)')
    #ax.coastlines()
    #ax.add_feature(cfeature.BORDERS, edgecolor='grey', alpha=0.7)