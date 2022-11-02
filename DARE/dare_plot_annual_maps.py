#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 16:45:00 2022

Plot emission maps and difference maps

@author: mlunt
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colorbar as cbar
import json

def open_ds(fname):
    with xarray.open_dataset(fname) as ds:
        ds.load()
    return ds

domain = "DARE"
post_dir = "/home/mlunt/datastore/DARE/inv_outputs/inv_runs/emissions/new_emis/" 
post_dir2 = "/home/mlunt/datastore/DARE/inv_outputs/inv_runs/emissions/new_emis/" 

# Inversions to include

#version1 = "co2_only_20pcap"
#version1 = "paper_co2_only"
#version2 = "paper_coco2"
version1 = "co2_only_newBC"
version2 = "coco2_newBC"
#version1 = "coco2_newBC_no_BD_HF_TC"


fname_in1 = post_dir +  "post_new_emissions_" + version1 + ".nc"
fname_in2 = post_dir2 +  "post_new_emissions_" + version2 + ".nc"

ds1=open_ds(fname_in1)
ds2=open_ds(fname_in2)


#%%
# Plot map to show where emission differences are between 20% uncert and 20% no MHD
emis_ap_co2sum = ds2.emis_ap_co2ff + ds2.emis_ap_co2bio  #+ ds2.emis_ap_co2reco + ds2.emis_ap_co2gee
emis_ap_co2bio  = ds2.emis_ap_co2bio   #(ds2.emis_ap_co2reco + ds2.emis_ap_co2gee)
emis_ap_co2ff = ds1.emis_ap_co2ff
emis_ap_coff = ds1.emis_ap_coff

emis1_co2sum = ds1.emis_mean_co2ff + ds1.emis_mean_co2bio   #ds1.emis_mean_co2reco + ds1.emis_mean_co2gee
emis2_co2sum = ds2.emis_mean_co2ff + ds2.emis_mean_co2bio  #+ ds2.emis_mean_co2reco + ds2.emis_mean_co2gee
 
emis1_co2bio  =  ds1.emis_mean_co2bio  #(ds1.emis_mean_co2reco + ds1.emis_mean_co2gee)
emis2_co2bio  = ds2.emis_mean_co2bio #(ds2.emis_mean_co2reco + ds2.emis_mean_co2gee)

emis1_co2ff  = ds1.emis_mean_co2ff
emis2_co2ff  = ds2.emis_mean_co2ff

emis1_coff  = ds1.emis_mean_coff
emis2_coff  = ds2.emis_mean_coff

emis1_co2sum_an = emis1_co2sum.groupby('time.year').mean(dim="time")
emis2_co2sum_an = emis2_co2sum.groupby('time.year').mean(dim="time")
emis_ap_co2sum_an = emis_ap_co2sum.groupby('time.year').mean(dim="time")

emis1_co2bio_an = emis1_co2bio.groupby('time.year').mean(dim="time")
emis2_co2bio_an = emis2_co2bio.groupby('time.year').mean(dim="time")
emis_ap_co2bio_an = emis_ap_co2bio.groupby('time.year').mean(dim="time")

emis1_co2ff_an = emis1_co2ff.groupby('time.year').mean(dim="time")
emis2_co2ff_an = emis2_co2ff.groupby('time.year').mean(dim="time")
emis_ap_co2ff_an = emis_ap_co2ff.groupby('time.year').mean(dim="time")

emis1_coff_an = emis1_coff.groupby('time.year').mean(dim="time")
emis2_coff_an = emis2_coff.groupby('time.year').mean(dim="time")
emis_ap_coff_an = emis_ap_coff.groupby('time.year').mean(dim="time")


lon = ds1.lon.values
lat = ds1.lat.values
dlat = lat[1] - lat[0]
dlon = lon[1] - lon[0]

#%%

#map_data = emis2_co2sum.mean(dim="time") - emis_ap_co2sum.mean(dim="time") 
#map_data = emis2_co2sum.mean(dim="time") - emis1_co2sum.mean(dim="time") 

cmin = -2.e-8
cmax = 2.e-8
proj = ccrs.PlateCarree()


#    
#%%
# Plot seasonal map each year 
# First calculate seasonal mean
emis2_co2sum_season = emis2_co2sum.groupby('time.season').mean(dim="time")
emis2_co2_seas_anom = emis2_co2sum.groupby('time.season') - emis2_co2sum_season
emis2_co2sum_qs = emis2_co2_seas_anom.resample(time="QS-DEC").mean(dim="time")

jja_data = emis2_co2sum_qs[2::4,:,:]
map_data=[emis2_co2sum_season.sel(season="JJA")]
for xi in range(3):
    map_data.append(jja_data[xi])


#proj = ccrs.PlateCarree()
#fig,axes = plt.subplots(2,2,subplot_kw=dict(projection=proj),figsize=(8,6))
#axs= axes.ravel()
#for xi in range(4):
#    
#    #map_data2 = (emis3_co2sum_season - emis2_ap_co2sum_season).sel(season = season_list[xi])*60*60*24*365
#    #map_data2 = emis3_co2bio.sel(season = season_list[xi])*60*60*24*365
#    
#    map_data2 = map_data[xi]*60*60*24*365
#    
#    
#    if xi==0:
#        cmin=0
#        cmax = 5
#        cmap="viridis"
#    else:
#        cmin = -0.5
#        cmax=0.5
#        cmap = "RdBu_r"
#    
#    h2a1 = axs[xi].pcolormesh(lon-dlon/2., lat-dlat/2., map_data2,
#     transform=ccrs.PlateCarree(), cmap=cmap, vmin=cmin, vmax=cmax)
#    
#    axs[xi].coastlines()
#    axs[xi].add_feature(cfeature.BORDERS)
#    axs[xi].set_extent((-15,25, 40,64))
#    #axs[xi].set_title(season_list[xi])
#
##cbaxes2 = fig2.add_axes([0.82, 0.12, 0.02, 0.76]) 
#cbaxes2 = fig.add_axes([0.12, 0.09, 0.76, 0.02]) 
#cb2 = plt.colorbar(h2a1, cax = cbaxes2, orientation='horizontal', extend='both')
##cb2.set_label('Posterior NEE flux (kg m$^{-2}$ yr$^{-1}$)', size=11)
#cb2.set_label('CO$_2^{ff}$ flux difference (kg m$^{-2}$ yr$^{-1}$)', size=11)


#%%

cmin = -0.5
cmax = 0.5

year_list=["2017", "2018", "2019"]
alphabet = ["(a)", "(b)", "(c)", "(d)"]
proj = ccrs.PlateCarree()
fig,axes = plt.subplots(2,2,subplot_kw=dict(projection=proj),figsize=(8,6))
axs= axes.ravel()
for xi in range(3):
    
    #map_data2 = (emis3_co2sum_season - emis2_ap_co2sum_season).sel(season = season_list[xi])*60*60*24*365
    #map_data2 = emis3_co2bio.sel(season = season_list[xi])*60*60*24*365
    
    map_data2 = ((emis2_co2sum_an- emis_ap_co2sum_an)[xi,:,:])*60*60*24*365
    
    h2a1 = axs[xi].pcolormesh(lon-dlon/2., lat-dlat/2., map_data2,
     transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=cmin, vmax=cmax)
    
    axs[xi].coastlines()
    axs[xi].add_feature(cfeature.BORDERS)
    axs[xi].set_extent((-12.5,25, 40,64))
    axs[xi].set_title(year_list[xi])

map_data = ((emis2_co2sum_an- emis_ap_co2sum_an).mean(dim="year"))*60*60*24*365
h2a2 = axs[3].pcolormesh(lon-dlon/2., lat-dlat/2., map_data,
     transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=cmin, vmax=cmax)
axs[3].coastlines()
axs[3].add_feature(cfeature.BORDERS)
axs[3].set_extent((-12.5,25, 40,64))
axs[3].set_title("3-year mean")

#cbaxes2 = fig2.add_axes([0.82, 0.12, 0.02, 0.76]) 
cbaxes2 = fig.add_axes([0.12, 0.09, 0.76, 0.02]) 
cb2 = plt.colorbar(h2a1, cax = cbaxes2, orientation='horizontal', extend='both')
#cb2.set_label('Posterior NEE flux (kg m$^{-2}$ yr$^{-1}$)', size=11)
cb2.set_label('CO$_2^{net}$ flux difference (kg m$^{-2}$ yr$^{-1}$)', size=11)

for xi in range(4):
    axs[xi].text(-0.1,0.85, alphabet[xi], transform = axs[xi].transAxes, fontsize=11)
#for site in p_sites:
#    dict_si = site_info[site]
#    network_si = dict_si["network"]
#    
#    lat_si = dict_si["latitude"]
#    lon_si = dict_si["longitude"]
#    
#    ax.scatter(lon_si,lat_si, color='black', s=16)
#    ax.text(lon_si,lat_si, site)
#%%
# Plot maps for COff

cmin = -8e-3
cmax = 8e-3

proj = ccrs.PlateCarree()
fig,axes = plt.subplots(2,2,subplot_kw=dict(projection=proj),figsize=(8,6))
axs= axes.ravel()
for xi in range(3):
    
    map_data3 = ((emis2_coff_an - emis_ap_coff_an)[xi,:,:])*60*60*24*365
    #map_data3 = (emis2_coff_season - emis1_coff_season).sel(season = season_list[xi])*60*60*24*365
    
    
    h2a1 = axs[xi].pcolormesh(lon-dlon/2., lat-dlat/2., map_data3,
     transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=cmin, vmax=cmax)
    
    axs[xi].coastlines()
    axs[xi].add_feature(cfeature.BORDERS)
    axs[xi].set_extent((-15,25, 40,64))
    #axs[xi].set_title(season_list[xi])

#cbaxes2 = fig2.add_axes([0.82, 0.12, 0.02, 0.76]) 
cbaxes2 = fig.add_axes([0.12, 0.09, 0.76, 0.02]) 
cb2 = plt.colorbar(h2a1, cax = cbaxes2, orientation='horizontal', extend='both')
#cb2.set_label('Posterior NEE flux (kg m$^{-2}$ yr$^{-1}$)', size=11)
cb2.set_label('CO$^{ff}$ flux difference (kg m$^{-2}$ yr$^{-1}$)', size=11)
