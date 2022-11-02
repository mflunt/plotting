#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:41:31 2022

Plot figure 3 for EAF paper

CHris wants the time lag of emissions since the OND IOD peak. 

i.e. months since Oct 2019 till peak emissions
Does this follow the water? 

At least 3 regions, SE, NE (Sobat) and Sudd.

Include TROPOMI data (and GOSAT in this comparison)

@author: mlunt
"""

import numpy as np
import xarray
import areagrid
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colorbar as cbar
import glob
import matplotlib.colors as colors
import matplotlib.patches as mpatches

def open_ds(fname, group=None):
    with xarray.open_dataset(fname, group) as ds:
        ds.load()
    return ds

# EAF inversion outputs
#version1 = "eaf_transd_gosat_lst2"
version1 = "eaf_transd_gs_lst2_v2cs"
#version1 = "eaf_transd_gosat_wetcharts"
#version2 = "eaf_transd_tropomi_lst2"
post_dir = "/home/mlunt/datastore/EAF/inv_outputs/paper/emissions/" 

fname_inv = post_dir +  "post_emissions_gosat_ensemb10.nc"

#fname_inv = post_dir +  "post_emissions_" + version1 + ".nc"
#fname_inv = post_dir +  "post_emissions_" + version2 + ".nc"

ds_inv = open_ds(fname_inv)

#%%
# Read in Chris' LST and plot the same
chris_dir = "/home/mlunt/datastore/EAF/emissions/"
fname_chris100 = chris_dir + "ch4_lst_wetland_flux_merged_v2_Cs_100cm_025x025.nc"

ds_chris_temp100 = open_ds(fname_chris100)
ds_chris100 = ds_chris_temp100.sel(lon=slice(20,55),
               lat=slice(-12, 20), time=slice("20191001", "20211231"))

cflux0_clim = ds_chris_temp100.flux0.groupby('time.month').mean(dim="time")
cflux0_anom_all = ds_chris_temp100.flux0.groupby('time.month') - cflux0_clim

cflux0_anom_temp = cflux0_anom_all.sel(lon=slice(20,55),
               lat=slice(-12, 20), time=slice("20191001", "20211231"))

cflux0_anom = cflux0_anom_temp.fillna(0.).resample(time="QS-OCT").mean(dim="time")

cflux0 = ds_chris100.flux0.fillna(0.).resample(time="QS-OCT").mean(dim="time")
cflux2 = ds_chris100.flux2.fillna(0.).resample(time="QS-OCT").mean(dim="time")

lat_lst  = cflux2.lat.values
lon_lst  = cflux2.lon.values

cflux2_max = cflux2.max(dim="time")
lst2_month_max_temp = cflux2.argmax(dim="time")
lst2_month_max = lst2_month_max_temp.where(cflux2_max>0)

cflux0_max = cflux0.max(dim="time")
lst0_month_max_temp = cflux0.argmax(dim="time")
lst0_month_max = lst0_month_max_temp.where(cflux0_max>0)

cflux0_anom_max = cflux0_anom.max(dim="time")
lst0_anom_month_max_temp = cflux0_anom.argmax(dim="time")
lst0_anom_month_max = lst0_anom_month_max_temp.where(cflux0_anom_max>0)

#%%
# Read in TROPOMI and GOSAT data
fname_trop = "/home/mlunt/datastore/MOYA/africa/" + "TROPOMI_EAF_01x01_monthly.nc"
fname_gosat = "/home/mlunt/datastore/MOYA/africa/GOSAT_EAF_05x05_monthly.nc"

ds_trop = open_ds(fname_trop)

trop_plus = ds_trop.XCH4_enhancement.sel(lon=slice(20,55),
               lat=slice(-12, 20), time=slice("20191001", "20211231"))

trop_plus2 = trop_plus.fillna(0.).resample(time="QS-OCT").mean(dim="time")
                                         
trop_max = trop_plus2.max(dim="time")
trop_month_max_temp = trop_plus2.argmax(dim="time")
trop_month_max = trop_month_max_temp.where(trop_max>0)

lon_trop = trop_plus.lon.values
lat_trop = trop_plus.lat.values

ds_gosat = open_ds(fname_gosat)

gosat_plus = ds_gosat.CH4_enhancement.sel(lon=slice(20,55),
               lat=slice(-12, 20), time=slice("20191001", "20211231"))

gosat_plus2 = gosat_plus.fillna(0.).resample(time="QS-OCT").mean(dim="time")
                                         
gosat_max = gosat_plus2.max(dim="time")
gosat_month_max_temp = gosat_plus2.argmax(dim="time")
gosat_month_max = gosat_month_max_temp.where(gosat_max>0)

lon_gosat = gosat_plus.lon.values
lat_gosat = gosat_plus.lat.values
#%%
# Calculate for each grid cell where maximum emissions are
emis_ens = ds_inv.emis_ens
emis_all = emis_ens.mean(dim="ens")
emis_clim = emis_all.groupby('time.month').mean(dim="time")
emis_anom_all = emis_all.groupby('time.month') - emis_clim

emis = emis_all.sel(time=slice("20191001", "20211231")).resample(time="QS-OCT").mean(dim="time")
emis_anom = emis_anom_all.sel(time=slice("20191001", "20211231")).resample(time="QS-OCT").mean(dim="time")


emis_max = emis.max(dim="time")
emis_anom_max = emis_anom.max(dim="time")

lat = emis.lat.values
lon = emis.lon.values
nlat = len(lat)
nlon=len(lon)

#month_max = emis_max.copy()*0.
#month_max = month_max.where(month_max>0)
#for lai in range(nlat):
#    for loi in range(nlon):
#        
#        if emis_max[lai,loi]>0:
#            wh_max = np.where(emis[:,lai,loi] == emis_max)
#
#            month_max[lai,loi] = wh_max[0][0]

month_max_temp = emis.argmax(dim="time")
month_max = month_max_temp.where(emis_max>0)

anom_month_max_temp = emis_anom.argmax(dim="time")
anom_month_max = anom_month_max_temp.where(emis_anom_max>0)

#%%
# Pick out sobat, sudd and pibor - overplot on figures below for sanity check

sud_lons = [28.5,31.]
sud_lats = [6.25,9.75]

sob_lons = [31.,33.5]
sob_lats = [7.8,11.]

pib_lons = [31.,33.5]
pib_lats = [4.75,7.8]

area = areagrid.areagrid(lat,lon)
area_chris = areagrid.areagrid(lat_lst,lon_lst)

emis_chris = ds_chris_temp100.flux2.fillna(0.).sel(time=slice("20190101", "20211231"))

emis_area = emis_chris*area_chris
#emis_area = emis_all*area
t_convert = 60*60*24*265/1.e9

sud_mean = emis_area.sel(lon=slice(sud_lons[0],sud_lons[1]), 
                        lat=slice(sud_lats[0],sud_lats[1])).sum(dim=["lat","lon"])*t_convert

sob_mean = emis_area.sel(lon=slice(sob_lons[0],sob_lons[1]), 
                        lat=slice(sob_lats[0],sob_lats[1])).sum(dim=["lat","lon"])*t_convert

pib_mean = emis_area.sel(lon=slice(pib_lons[0],pib_lons[1]), 
                        lat=slice(pib_lats[0],pib_lats[1])).sum(dim=["lat","lon"])*t_convert
                         
sud_clim = sud_mean.groupby('time.month').mean(dim="time")
sob_clim = sob_mean.groupby('time.month').mean(dim="time")
pib_clim = pib_mean.groupby('time.month').mean(dim="time")

sud_anom = sud_mean.groupby('time.month') - sud_clim
sob_anom = sob_mean.groupby('time.month') - sob_clim
pib_anom = pib_mean.groupby('time.month') - pib_clim

#%%
cmin=0
cmax = 27
bounds = np.arange(cmin,cmax+3,3)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
dlat = 0.25
dlon=0.3125
proj = ccrs.PlateCarree()
fig2,ax2=plt.subplots(subplot_kw=dict(projection=proj),figsize=(8,8))
h2 = ax2.pcolormesh(lon-dlon/2., lat-dlat/2., anom_month_max*3, transform=ccrs.PlateCarree(), cmap='Spectral', norm=norm)
#h2 = ax2.pcolormesh(lon1-dlon1/2., lat1-dlat1/2., map_data, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-4, vmax=4)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS)


cax2,kw2 = cbar.make_axes(ax2,location='bottom',pad=0.01,shrink=0.7)
out=fig2.colorbar(h2,cax=cax2,**kw2, label="Months since October 2019")            

ax2.set_extent((25,40,-2,15))

ax2.add_patch(mpatches.Rectangle(xy = [sud_lons[0], sud_lats[0]], 
                                 width = sud_lons[1] - sud_lons[0],
                                 height = sud_lats[1] - sud_lats[0], 
                                 transform = ccrs.PlateCarree(),
                                 facecolor = "None", edgecolor="black"))

ax2.add_patch(mpatches.Rectangle(xy = [sob_lons[0], sob_lats[0]], 
                                 width = sob_lons[1] - sob_lons[0],
                                 height = sob_lats[1] - sob_lats[0], 
                                 transform = ccrs.PlateCarree(),
                                 facecolor = "None", edgecolor="black"))

ax2.add_patch(mpatches.Rectangle(xy = [pib_lons[0], pib_lats[0]], 
                                 width = pib_lons[1] - pib_lons[0],
                                 height = pib_lats[1] - pib_lats[0], 
                                 transform = ccrs.PlateCarree(),
                                 facecolor = "None", edgecolor="black"))
#%%
proj = ccrs.PlateCarree()
fig2,ax2=plt.subplots(subplot_kw=dict(projection=proj),figsize=(8,8))
h2 = ax2.pcolormesh(lon_lst, lat_lst, lst0_anom_month_max*3, transform=ccrs.PlateCarree(), cmap='Spectral', norm=norm)
#h2 = ax2.pcolormesh(lon1-dlon1/2., lat1-dlat1/2., map_data, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-4, vmax=4)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS)


cax2,kw2 = cbar.make_axes(ax2,location='bottom',pad=0.01,shrink=0.7)
out=fig2.colorbar(h2,cax=cax2,**kw2, label="Months since October 2019")      
ax2.set_extent((25,40,-2,15))      

ax2.add_patch(mpatches.Rectangle(xy = [sud_lons[0], sud_lats[0]], 
                                 width = sud_lons[1] - sud_lons[0],
                                 height = sud_lats[1] - sud_lats[0], 
                                 transform = ccrs.PlateCarree(),
                                 facecolor = "None", edgecolor="black"))

ax2.add_patch(mpatches.Rectangle(xy = [sob_lons[0], sob_lats[0]], 
                                 width = sob_lons[1] - sob_lons[0],
                                 height = sob_lats[1] - sob_lats[0], 
                                 transform = ccrs.PlateCarree(),
                                 facecolor = "None", edgecolor="black"))

ax2.add_patch(mpatches.Rectangle(xy = [pib_lons[0], pib_lats[0]], 
                                 width = pib_lons[1] - pib_lons[0],
                                 height = pib_lats[1] - pib_lats[0], 
                                 transform = ccrs.PlateCarree(),
                                 facecolor = "None", edgecolor="black"))

#%%

#Plot tropomi month max

proj = ccrs.PlateCarree()
fig2,ax2=plt.subplots(subplot_kw=dict(projection=proj),figsize=(8,8))
h2 = ax2.pcolormesh(lon_trop, lat_trop, trop_month_max*3, transform=ccrs.PlateCarree(), cmap='Spectral', norm=norm)
#h2 = ax2.pcolormesh(lon1-dlon1/2., lat1-dlat1/2., map_data, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-4, vmax=4)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS)


cax2,kw2 = cbar.make_axes(ax2,location='bottom',pad=0.01,shrink=0.7)
out=fig2.colorbar(h2,cax=cax2,**kw2, label="Months since October 2019")      
ax2.set_extent((25,40,-2,15))      

ax2.add_patch(mpatches.Rectangle(xy = [sud_lons[0], sud_lats[0]], 
                                 width = sud_lons[1] - sud_lons[0],
                                 height = sud_lats[1] - sud_lats[0], 
                                 transform = ccrs.PlateCarree(),
                                 facecolor = "None", edgecolor="black"))

ax2.add_patch(mpatches.Rectangle(xy = [sob_lons[0], sob_lats[0]], 
                                 width = sob_lons[1] - sob_lons[0],
                                 height = sob_lats[1] - sob_lats[0], 
                                 transform = ccrs.PlateCarree(),
                                 facecolor = "None", edgecolor="black"))

ax2.add_patch(mpatches.Rectangle(xy = [pib_lons[0], pib_lats[0]], 
                                 width = pib_lons[1] - pib_lons[0],
                                 height = pib_lats[1] - pib_lats[0], 
                                 transform = ccrs.PlateCarree(),
                                 facecolor = "None", edgecolor="black"))


#%%
proj = ccrs.PlateCarree()
fig2,ax2=plt.subplots(subplot_kw=dict(projection=proj),figsize=(8,8))
h2 = ax2.pcolormesh(lon_gosat, lat_gosat, gosat_month_max*3, transform=ccrs.PlateCarree(), cmap='Spectral', norm=norm)
#h2 = ax2.pcolormesh(lon1-dlon1/2., lat1-dlat1/2., map_data, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-4, vmax=4)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS)


cax2,kw2 = cbar.make_axes(ax2,location='bottom',pad=0.01,shrink=0.7)
out=fig2.colorbar(h2,cax=cax2,**kw2, label="Months since October 2019")      
ax2.set_extent((25,40,-2,15))      

ax2.add_patch(mpatches.Rectangle(xy = [sud_lons[0], sud_lats[0]], 
                                 width = sud_lons[1] - sud_lons[0],
                                 height = sud_lats[1] - sud_lats[0], 
                                 transform = ccrs.PlateCarree(),
                                 facecolor = "None", edgecolor="black"))

ax2.add_patch(mpatches.Rectangle(xy = [sob_lons[0], sob_lats[0]], 
                                 width = sob_lons[1] - sob_lons[0],
                                 height = sob_lats[1] - sob_lats[0], 
                                 transform = ccrs.PlateCarree(),
                                 facecolor = "None", edgecolor="black"))

ax2.add_patch(mpatches.Rectangle(xy = [pib_lons[0], pib_lats[0]], 
                                 width = pib_lons[1] - pib_lons[0],
                                 height = pib_lats[1] - pib_lats[0], 
                                 transform = ccrs.PlateCarree(),
                                 facecolor = "None", edgecolor="black"))

#%%
# Plot Sudd,Sobat and Pibor emissions
wetlands = ["Sudd", "Sobat", "Pibor"]

riv_emis_ens = ds_inv.ssud_ens.sel(time=slice("20190101", "20211231")) 
riv_emis_mean = riv_emis_ens.mean(dim="ens")

fig,axs = plt.subplots(2, sharex=True)
for ti in range(3):
    axs[0].plot(riv_emis_mean.time, riv_emis_mean[ti,:], label = wetlands[ti])

axs[1].plot(sud_mean.time, sud_mean)
axs[1].plot(sud_mean.time, sob_mean)
axs[1].plot(sud_mean.time, pib_mean)

axs[0].set_ylabel("Emissions rate (Tg yr$^{-1}$)")
axs[1].set_ylabel("Emissions rate (Tg yr$^{-1}$)")
axs[0].legend()

axs[0].set_title("Inversion output")
axs[1].set_title("LST 2K threshold 100cm")

fig.autofmt_xdate()