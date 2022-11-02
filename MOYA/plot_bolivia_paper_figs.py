#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:37:07 2020

Plot bolivia flux outputs from mine and Anita's results

@author: mlunt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colorbar as cbar
from cartopy.io.img_tiles import Stamen as OSM
import xarray
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#from acrg_grid import haversine
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import scalebar_mod
import matplotlib.dates as mdates
import glob

def open_ds(fname):
    with xarray.open_dataset(fname) as ds:
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

tiler = OSM(style='terrain-background')
mercator = tiler.crs


fname_gc = "/home/mlunt/datastore/MOYA/bolivia/bolivia_inversion_outputs.nc"
ds_gc = open_ds(fname_gc)

#fname_name = "/home/mlunt/datastore/MOYA/bolivia/CH4_SOUTHAMERICA_swamps_2019-03-08.nc"

fname_name = "/home/mlunt/datastore/MOYA/bolivia/CH4_SOUTHAMERICA-0.14-0.09_flat-truncatednormal-sd5-2000basis_2019-03-08_reduced.nc"
ds_name = open_ds(fname_name)

#%%
# Read in global model background file
gosat_dir = "/home/mlunt/ceph/verify/model_settings/DARE_runs/CH4/GOSAT/"
fname_gosat = gosat_dir + "XCH4_scaled_2x25_GOSAT_2010_2020.nc"

ds_gosat = open_ds(fname_gosat)

xch4_bg_bol = ds_gosat.XCH4_BG_BOL
#xch4_bg_bol2 = xch4_bg_bol.reindex_like(xch4, method="nearest")

xch4_bol_mod = ds_gosat.XCH4_BOL
#xch4_bol_mod_zonal = xch4_bol_mod2.mean(dim="lon")
#xch4_bol_mod = xch4_bol_mod2 - xch4_bol_mod_zonal # Subtract global growth int this term to only get local enhancement

#xch4_bol_mod_05 = xch4_bol_mod.reindex_like(xch4, method="nearest")

#xch4_anom = xch4 - xch4_bg_bol2
xch4_anom = ds_gosat.XCH4_obs - xch4_bg_bol -xch4_bol_mod

#%%
lonmin = -66.5
lonmax = -64.5
latmin = -15
latmax= -13

#xch4_diff = (xch4_anom + xch4_bol_mod).sel(lon=slice(-75,-50), lat=slice(-20,0) )
xch4_diff = (xch4_anom).sel(lon=slice(-75,-50), lat=slice(-20,0) )
xch4_diff_mn = xch4_diff.groupby('time.month').mean(dim="time")
map_data = xch4_diff_mn[:3,:,:].mean(dim="month")

#map_clim = (xch4_anom).sel(lon=slice(-75,-50), lat=slice(-20,0)).groupby('time.month').mean(dim="time")
#map_data = map_clim[:3,:,:].mean(dim="month")

#map_data= (xch4_anom + xch4_bol_mod).sel(lon=slice(-75,-50), lat=slice(-20,0)).mean(dim="time")
bounds0 = np.linspace(-32, 32, 17)
norm0 = colors.BoundaryNorm(boundaries=bounds0, ncolors=256)
# For paper plot difference between data and model 

lon_map=map_data.lon.values
lat_map = map_data.lat.values
dlon = (lon_map[1]-lon_map[0])
dlat = (lat_map[1]-lat_map[0])

proj = ccrs.PlateCarree()
    
fig3,ax3=plt.subplots(subplot_kw=dict(projection=proj),figsize=(8,8))
h3 = ax3.pcolormesh(lon_map-dlon/2, lat_map-dlat/2, map_data, transform=ccrs.PlateCarree(), cmap='RdBu_r', norm=norm0)
ax3.coastlines()
ax3.add_feature(cfeature.BORDERS)
cax3,kw3 = cbar.make_axes(ax3,location='bottom',pad=0.08,shrink=0.7)
cb3=fig3.colorbar(h3,cax=cax3,extend='both',**kw3, label="GOSAT XCH$_4$ - model XCH$_4$ (ppb)")

cb3_ticks = [-32, -16,0, 16,32]
#cb4_ticks = [0,8,16,24,32,40]
cb3.set_ticks([cb3_ticks])

#ax3.add_patch(mpatches.Rectangle(xy=[lonmin, latmin], 
#                                 width=lonmax-lonmin, 
#                                 height=latmax - latmin,
#                                    fill=False, edgecolor="black",
#                                    alpha=1,
#                                    transform=ccrs.PlateCarree()))

gl = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = mticker.FixedLocator([-85,-80,-75,-70, -65, -60, -55, -45])
gl.ylocator = mticker.FixedLocator([-28,-24, -20,-16,-12, -8,-4,0,8])#            "20180701", "20180716",
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

#%%
y_obs_gc = ds_gc.ch4_observed

y_post_gc2 = ds_gc.ch4_model_post
y_ap_gc2 = ds_gc.ch4_model_prior

y_post_gc = ds_gc.ch4_model_wetcharts_post
y_ap_gc = ds_gc.ch4_model_wetcharts_prior

y_lat_gc = ds_gc.faam_lat
y_lon_gc = ds_gc.faam_lon
y_time_gc = ds_gc.time
y_post_gcA = (y_post_gc.where(y_time_gc.dt.day ==8, drop=True)).resample(time='1min').mean()
y_post_gcB = (y_post_gc.where(y_time_gc.dt.day ==9, drop=True)).resample(time='1min').mean()

y_ap_gcA = (y_ap_gc.where(y_time_gc.dt.day ==8, drop=True)).resample(time='1min').mean()
y_ap_gcB = (y_ap_gc.where(y_time_gc.dt.day ==9, drop=True)).resample(time='1min').mean()

y_time_gcA = y_post_gcA.time
y_time_gcB = y_post_gcB.time

y_obsA = y_obs_gc.where(y_time_gc.dt.day ==8, drop=True)
y_obsB = y_obs_gc.where(y_time_gc.dt.day ==9, drop=True)

lon_gc = ds_gc.lon.values
lat_gc = ds_gc.lat.values

x_post_gc2 = ds_gc.flux_post
x_ap_gc2 = ds_gc.flux_prior

x_post_gc = ds_gc.flux_wetcharts_post
x_ap_gc = ds_gc.flux_wetcharts_prior

x_uncert_gc = ds_gc.uncertainty_wetcharts_post
#%%
y_obs_name2 = ds_name.Yobs
y_post_name2 = ds_name.Ymodmean
y_ap_name2 = ds_name.Yapriori
#y_time_name = ds_name.Ytime

ytime_name = ds_name.Ytime.values

ds_name_obs = xarray.Dataset({"y_obs": (["time"], y_obs_name2.values),
                          "y_post": (["time"], y_post_name2.values),
                          "y_ap": (["time"], y_ap_name2.values),},
    coords={"time": ytime_name})

ds_name_obs2 = ds_name_obs.reindex_like(y_obs_gc, method = "nearest")

y_obs_name = ds_name_obs2.y_obs
y_post_name = ds_name_obs2.y_post
y_ap_name = ds_name_obs2.y_ap
y_time_name = ds_name_obs2.time

y_lat_name = y_lat_gc.copy()
y_lon_name = y_lon_gc.copy()

lon_name = ds_name.lon.values
lat_name = ds_name.lat.values

x_post_name = ds_name.fluxmean*16.04*60.*60.*1000. #mol/m2/s to mg/m2/hour
x_ap_name = ds_name.fluxapriori*16.04*60.*60.*1000.

y_post_nameA = (y_post_name.where(y_time_name.dt.day ==8, drop=True)).resample(time='1min').mean()
y_post_nameB = (y_post_name.where(y_time_name.dt.day ==9, drop=True)).resample(time='1min').mean()

y_ap_nameA = (y_ap_name.where(y_time_name.dt.day ==8, drop=True)).resample(time='1min').mean()
y_ap_nameB = (y_ap_name.where(y_time_name.dt.day ==9, drop=True)).resample(time='1min').mean()

y_time_nameA = y_time_name.where(y_time_name.dt.day ==8, drop=True)
y_time_nameB = y_time_name.where(y_time_name.dt.day ==9, drop=True)

y_obs_nameA = (y_obs_name.where(y_time_name.dt.day==8,drop=True))
y_obs_nameB = (y_obs_name.where(y_time_name.dt.day==9,drop=True))

y_lon_nameA = (y_lon_name.where(y_time_name.dt.day==8,drop=True))
y_lon_nameB = (y_lon_name.where(y_time_name.dt.day==9, drop=True))

y_lat_nameA = (y_lat_name.where(y_time_name.dt.day==8, drop=True))
y_lat_nameB = (y_lat_name.where(y_time_name.dt.day==9, drop=True))

#%%

proj = ccrs.PlateCarree()

obs_min = 1880
obs_max = 2300

cmax=obs_max
cmin=obs_min
bounds = np.linspace(cmin, cmax, 11)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

cmax2=12
cmin2=0
bounds2 = np.linspace(cmin2, cmax2, 11)
norm2 = colors.BoundaryNorm(boundaries=bounds2, ncolors=256)

fig=plt.figure(figsize=(9,10))

ax1 = plt.subplot2grid((3, 2), (0, 1), projection=proj)
ax2 = plt.subplot2grid((3, 2), (1, 0), projection=proj)
ax3 = plt.subplot2grid((3, 2), (1, 1), projection=proj)
ax4 = plt.subplot2grid((3, 2), (2, 0), projection=proj)
ax5 = plt.subplot2grid((3, 2), (2, 1), projection=proj)



#fig,axes=plt.subplots(3,2, subplot_kw=dict(projection=proj),figsize=(9,10),sharex=True, sharey=True)
#axs=axes.ravel()
map_extents = [np.min(y_lon_gc)-0.5, np.max(y_lon_gc)+2., np.min(y_lat_gc)-0.5, np.max(y_lat_gc)+0.5]


lon_gc2 = lon_gc-0.3125/2.
lat_gc2 = lat_gc-0.25/2.

dlon_name = lon_name[2]-lon_name[1]
dlat_name = lat_name[2]-lat_name[1]


lon_name2 = lon_name-dlon_name/2.
lat_name2 = lat_name-dlat_name/2.

rivers_50m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '50m')


axes=[ax1,ax2,ax3,ax4,ax5]

for axi in axes:
    axi.set_extent(map_extents, crs=ccrs.PlateCarree())
    #if ti >=2:
    axi.add_image(tiler,7)
    axi.add_feature(rivers_50m, facecolor='None', edgecolor='b')

#axs[0].pcolormesh(lon_riv, lat_riv, river2,
#                transform=ccrs.PlateCarree(), cmap='RdPu', vmin=0.,vmax=0.5, alpha=0.5 )

#axs[0].add_feature(rivers_50m, facecolor='None', edgecolor='b')

scatter_map = 'viridis'

#h3 = axs[0].scatter(y_lon_gc, y_lat_gc, c=y_obs_gc, transform=ccrs.PlateCarree(), cmap=scatter_map, s=12, norm=norm)
#axs[0].set_title("Observations")

#axs[1].pcolormesh(lon_riv, lat_riv, river2,
#                transform=ccrs.PlateCarree(), cmap='RdPu', vmin=0.,vmax=0.5, alpha=0.5 )
h3 = ax1.scatter(y_lon_gc, y_lat_gc, c=y_obs_name, transform=ccrs.PlateCarree(), cmap=scatter_map, s=12, norm=norm)
ax1.set_title("Observations")

ax2.pcolormesh(lon_gc2, lat_gc2, x_ap_gc,
                transform=ccrs.PlateCarree(), cmap='RdPu', norm=norm2, alpha=0.5 )

h3 = ax2.scatter(y_lon_gc, y_lat_gc, c=y_ap_gc, transform=ccrs.PlateCarree(), cmap=scatter_map, s=12, norm=norm)
ax2.set_title("GEOS-Chem prior")

ax3.pcolormesh(lon_name, lat_name, x_ap_name,
                transform=ccrs.PlateCarree(), cmap='RdPu', norm=norm2, alpha=0.5 )
h3 = ax3.scatter(y_lon_gc, y_lat_gc, c=y_ap_name, transform=ccrs.PlateCarree(), cmap=scatter_map, s=12, norm=norm)
ax3.set_title("NAME prior")


ax4.pcolormesh(lon_gc2, lat_gc2, x_post_gc,
                transform=ccrs.PlateCarree(), cmap='RdPu', norm=norm2, alpha=0.5 )

h3 = ax4.scatter(y_lon_gc, y_lat_gc, c=y_post_gc, transform=ccrs.PlateCarree(), cmap=scatter_map, s=12, norm=norm)
ax4.set_title("GEOS-Chem posterior")

p3 = ax5.pcolormesh(lon_name, lat_name, x_post_name,
                transform=ccrs.PlateCarree(), cmap='RdPu', norm=norm2, alpha=0.5 )
h3 = ax5.scatter(y_lon_gc, y_lat_gc, c=y_post_name, transform=ccrs.PlateCarree(), cmap=scatter_map, s=12, norm=norm)
ax5.set_title("NAME posterior")

for ti, axi in enumerate(axes):
    #axs[ti].add_feature(rivers_50m, facecolor='None', edgecolor='b')
    gl = axi.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-68,-66,-65, -64, -62, -60])
    gl.ylocator = mticker.FixedLocator([-17, -15,-14,-13, -12])#            "20180701", "20180716",
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    if ti in [0,1,2]:
        gl.xlabels_bottom = False
    #if ti in [2,3,4,5]:
    #    gl.xlabels_top = False
        
    
    
cbaxes = fig.add_axes([0.1, 0.06, 0.8, 0.02]) 
##[left, bottom, width, height],
cb = plt.colorbar(h3, cax = cbaxes, orientation='horizontal', extend='both', label = 'CH4 mole fraction (ppb)') 

cbaxes = fig.add_axes([0.9, 0.1, 0.02, 0.8]) 
##[left, bottom, width, height],
cb = plt.colorbar(p3, cax = cbaxes, orientation='vertical', extend='both', label = 'CH4 flux (mg m$^{-2}$ hour$^{-1}$)') 

#%%
# Calculate emissions from same region for NAME and GEOS-Chem

reg_lons = [-65.1, -64.8]
reg_lats = [-14.3, -13.2]

#reg_lons = [-65.1, -64.6]
#reg_lats = [-14.3, -13.2]

reg_mean_gc = x_post_gc.sel(lon=slice(reg_lons[0], reg_lons[1]), 
                            lat=slice(reg_lats[0], reg_lats[1]))

# Mean  = 16 +/- 2.

reg_uncert_gc = x_uncert_gc.sel(lon=slice(reg_lons[0], reg_lons[1]), 
                            lat=slice(reg_lats[0], reg_lats[1]))

reg_mean_name = x_post_name.sel(lon=slice(reg_lons[0], reg_lons[1]), 
                            lat=slice(reg_lats[0], reg_lats[1]))

#%%
fig2=plt.figure(figsize=(9,12))

# Add consistent map plot with inset

ax0 = plt.subplot2grid((3, 3), (0, 0), projection=proj)
ax1 = plt.subplot2grid((3, 3), (0, 1), projection=proj)
ax4 = plt.subplot2grid((3, 3), (0, 2), projection=proj)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
#ax4 = plt.subplot2grid((3, 2), (2, 0), projection=proj)
#ax5 = plt.subplot2grid((3, 2), (2, 1), projection=proj)

map_extents0 = [-73, -53., -7., -21.]
ax0.set_extent(map_extents0, crs=ccrs.PlateCarree())

tiler2 = OSM(style='terrain')
mercator2 = tiler2.crs
ax0.add_image(tiler2,4)
ax0.add_patch(mpatches.Rectangle(xy=[y_lon_gc.min()-0.5, y_lat_gc.min()-0.5], 
                                 width=y_lon_gc.max()-y_lon_gc.min()+1, 
                                 height=y_lat_gc.max()-y_lat_gc.min()+1,
                                    fill=False, edgecolor="gold",
                                    alpha=1,
                                    transform=ccrs.PlateCarree()))

scalebar_mod.scale_bar(ax0, (0.75, 0.7), 250)

map_extents = [np.min(y_lon_gc)-0.5, np.max(y_lon_gc)+2., np.min(y_lat_gc)-0.5, np.max(y_lat_gc)+0.5]

lon_gc2 = lon_gc-0.3125/2.
lat_gc2 = lat_gc-0.25/2.

rivers_50m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '50m')
axes=[ax1]

for axi in axes:
    axi.set_extent(map_extents, crs=ccrs.PlateCarree())
    #if ti >=2:
    axi.add_image(tiler,7)
    axi.add_feature(rivers_50m, facecolor='None', edgecolor='b')

scalebar_mod.scale_bar(ax1, (0.75, 0.7), 100)

scatter_map = 'viridis'


#h3 = ax1.scatter(y_lon_gc, y_lat_gc, c=y_obs_name, transform=ccrs.PlateCarree(), cmap=scatter_map, s=12, norm=norm)

h3 = ax1.scatter(y_lon_nameA, y_lat_nameA, c=y_obs_nameA, transform = ccrs.PlateCarree(), 
                  cmap=scatter_map, s=10, norm=norm, marker='o')

h32 = ax1.scatter(y_lon_nameB, y_lat_nameB, c=y_obs_nameB, transform = ccrs.PlateCarree(),
                  cmap=scatter_map, s=10, norm=norm, marker='o')

ax1.annotate("Flight A", xy = (y_lon_nameA[10], y_lat_name[10]),
             xycoords="data", arrowprops = dict(facecolor="black", shrink=0.05),
             xytext = (0.05,0.2), textcoords="axes fraction")

ax1.annotate("Flight B", xy = (y_lon_nameB[-20], y_lat_nameB[-20]),
             xycoords="data", arrowprops=dict(facecolor="black", shrink=0.05),
             xytext=(0.7,0.6), textcoords="axes fraction")


#ax1.set_title("Observations")

for ti, axi in enumerate(axes):
    #axs[ti].add_feature(rivers_50m, facecolor='None', edgecolor='b')
    gl = axi.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-68,-66,-65, -64, -62, -60])
    gl.ylocator = mticker.FixedLocator([-17, -15,-14,-13, -12])#            "20180701", "20180716",
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    
#cbaxes2 = fig2.add_axes([0.59, 0.65, 0.015, 0.23]) 
cbaxes2 = fig2.add_axes([0.415, 0.86, 0.2, 0.015]) 
##[left, bottom, width, height],
cb2 = plt.colorbar(h3, cax = cbaxes2, orientation='horizontal', extend='both')
cb2.ax.xaxis.set_ticks_position('top')
cb2.ax.set_title('CH$_4$ mole fraction (ppb)', fontsize=10)
cb2_ticks = [obs_min, (obs_min+obs_max)//2,obs_max]
cb2.set_ticks([cb2_ticks])

# Draw lines between 2 maps to show zoom box
xyA0 = (map_extents0[0],map_extents0[2])
xyB0 = (map_extents[0],map_extents[2])
xyB1 = (map_extents[0],map_extents[3])

con = mpatches.ConnectionPatch(xyA = xyB0, xyB=xyB0, coordsA="data", 
                               coordsB="data", axesA = ax1, axesB=ax0, color="gold" )
ax1.add_artist(con)

con2 = mpatches.ConnectionPatch(xyA = xyB1, xyB=xyB1, coordsA="data", 
                               coordsB="data", axesA = ax1, axesB=ax0, color="gold" )
ax1.add_artist(con2)

bounds4 = np.linspace(0, 24, 11)
norm4 = colors.BoundaryNorm(boundaries=bounds4, ncolors=256)

h4 = ax4.pcolormesh(lon_map-dlon/2, lat_map-dlat/2, map_data, transform=ccrs.PlateCarree(), cmap='viridis', norm=norm4)
ax4.coastlines()
ax4.add_feature(cfeature.BORDERS)
#cbaxes4 = fig2.add_axes([0.89, 0.65, 0.015, 0.23]) 

cbaxes4 = fig2.add_axes([0.69, 0.86, 0.2, 0.015]) 
cb4=plt.colorbar(h4,cax=cbaxes4,orientation = "horizontal", extend='both')
cb4.ax.xaxis.set_ticks_position('top')
cb4.ax.set_title("GOSAT XCH$_4$ - model XCH$_4$ (ppb)", fontsize=10)
cb4_ticks = [0, 10,20]
cb4.set_ticks([cb4_ticks])

#label="GOSAT XCH$_4$ - model XCH$_4$")

ax4.set_extent(map_extents0, crs=ccrs.PlateCarree())

gl4 = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl4.xlabels_top = False
gl4.ylabels_left = False
gl4.xlocator = mticker.FixedLocator([-85,-80,-75,-70, -65, -60, -55, -45])
gl4.ylocator = mticker.FixedLocator([-28,-24, -20,-16,-12, -8,-4,0,8])#            "20180701", "20180716",
gl4.xformatter = LONGITUDE_FORMATTER
gl4.yformatter = LATITUDE_FORMATTER

n = 3
pcolors = plt.cm.viridis(np.linspace(0,1,n))

ax2.plot(y_obsA.time,y_obsA, 'o', markersize=4, color="black", linewidth=2, label="Observations")
ax2.plot(y_time_gcA,y_post_gcA, 's', markersize=4, color=pcolors[1], linewidth=2, label="GEOS-Chem posterior")
ax2.plot(y_time_gcA,y_post_nameA, '^', markersize=4, color="C4", linewidth=2, label="NAME posterior")

ax2.plot(y_time_gcA,y_ap_gcA, linestyle='--', color="grey", label="GEOS-Chem prior")
ax2.plot(y_time_gcA,y_ap_nameA, linestyle=':', markersize=4, color="grey", label="NAME prior")
#ax.plot(y_post2, color="C3", linestyle="--", label="Posterior model (WetCHARTs)")
#ax.plot(H[:,0]*x_post[0], color="C1", linestyle=':', label="Posterior baseline")

ax3.plot(y_obsB.time,y_obsB, 'o', markersize=4, color="black", linewidth=2, label="Observations")
ax3.plot(y_time_gcB,y_post_gcB, 's', markersize=4, color=pcolors[1], linewidth=2, label="GEOS-Chem posterior")
ax3.plot(y_time_gcB,y_post_nameB, '^', markersize=4, color="C4", linewidth=2, label="NAME posterior")

ax3.plot(y_time_gcB,y_ap_gcB, linestyle='--', color="grey", label="GEOS-Chem prior")
ax3.plot(y_time_gcB,y_ap_nameB, linestyle=':', markersize=4, color="grey", label="NAME prior")

ax2.set_ylabel("CH$_4$ (ppb)")
ax3.set_ylabel("CH$_4$ (ppb)")

ax3.set_xlabel("Time (UTC)")
ax2.legend(ncol=2)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


txt0 = ax0.text(0.05,0.1, "A", transform=ax0.transAxes, fontsize=14)
txt1 = ax1.text(0.05,0.1, "B", transform=ax1.transAxes, fontsize=14)
txt4 = ax4.text(0.05,0.1, "C", transform=ax4.transAxes, fontsize=14, color="white")
txt2 = ax2.text(0.02,0.05, "D", transform=ax2.transAxes, fontsize=14)
txt3 = ax3.text(0.02,0.05, "E", transform=ax3.transAxes, fontsize=14)

#fig2.text(0.1,0.8, "B")
#fig2.text(0.1,0.8, "C")
#fig2.text(0.1,0.8, "D")
#plt.xticks(rotation=45)
#ax2.set_xticklabels(ax2.get_xticklabels(),rotation=45)
#ax3.set_xticklabels(ax3.get_xticklabels(),rotation=45)
#fig2.autofmt_xdate()
#plt.tight_layout()

ax0.set_aspect('auto')
ax1.set_aspect('auto')


#%%
fig2=plt.figure(figsize=(9,12))

# Add consistent map plot with inset

ax0 = plt.subplot2grid((3, 4), (0, 0), projection=proj, colspan=2)
ax1 = plt.subplot2grid((3, 4), (0, 2), projection=proj, colspan=2)
ax4 = plt.subplot2grid((3, 4), (1, 2), projection=proj, colspan=2, rowspan=2)
ax2 = plt.subplot2grid((3, 4), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
#ax4 = plt.subplot2grid((3, 2), (2, 0), projection=proj)
#ax5 = plt.subplot2grid((3, 2), (2, 1), projection=proj)

map_extents0 = [-73, -53., -7., -21.]
ax0.set_extent(map_extents0, crs=ccrs.PlateCarree())

tiler2 = OSM(style='terrain')
mercator2 = tiler2.crs
ax0.add_image(tiler2,4)
ax0.add_patch(mpatches.Rectangle(xy=[y_lon_gc.min()-0.5, y_lat_gc.min()-0.5], 
                                 width=y_lon_gc.max()-y_lon_gc.min()+1, 
                                 height=y_lat_gc.max()-y_lat_gc.min()+1,
                                    fill=False, edgecolor="gold",
                                    alpha=1,
                                    transform=ccrs.PlateCarree()))

scalebar_mod.scale_bar(ax0, (0.75, 0.7), 250)

map_extents = [np.min(y_lon_gc)-0.5, np.max(y_lon_gc)+2., np.min(y_lat_gc)-0.5, np.max(y_lat_gc)+0.5]

lon_gc2 = lon_gc-0.3125/2.
lat_gc2 = lat_gc-0.25/2.

rivers_50m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '50m')
axes=[ax1]

for axi in axes:
    axi.set_extent(map_extents, crs=ccrs.PlateCarree())
    #if ti >=2:
    axi.add_image(tiler,7)
    axi.add_feature(rivers_50m, facecolor='None', edgecolor='b')

scalebar_mod.scale_bar(ax1, (0.75, 0.7), 100)
scatter_map = 'viridis'

h3 = ax1.scatter(y_lon_nameA, y_lat_nameA, c=y_obs_nameA, transform = ccrs.PlateCarree(), 
                  cmap=scatter_map, s=10, norm=norm, marker='o')

h32 = ax1.scatter(y_lon_nameB, y_lat_nameB, c=y_obs_nameB, transform = ccrs.PlateCarree(),
                  cmap=scatter_map, s=10, norm=norm, marker='o')

ax1.annotate("Flight A", xy = (y_lon_nameA[10], y_lat_name[10]),
             xycoords="data", arrowprops = dict(facecolor="black", shrink=0.05),
             xytext = (0.05,0.2), textcoords="axes fraction")

ax1.annotate("Flight B", xy = (y_lon_nameB[-20], y_lat_nameB[-20]),
             xycoords="data", arrowprops=dict(facecolor="black", shrink=0.05),
             xytext=(0.7,0.6), textcoords="axes fraction")

for ti, axi in enumerate(axes):
    #axs[ti].add_feature(rivers_50m, facecolor='None', edgecolor='b')
    gl = axi.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-68,-66,-65, -64, -62, -60])
    gl.ylocator = mticker.FixedLocator([-17, -15,-14,-13, -12])#            "20180701", "20180716",
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

cbaxes2 = fig2.add_axes([0.89, 0.65, 0.015, 0.23]) 
#cbaxes2 = fig2.add_axes([0.415, 0.86, 0.2, 0.015]) 
##[left, bottom, width, height],
cb2 = plt.colorbar(h3, cax = cbaxes2, orientation='vertical', extend='both')
#cb2.ax.xaxis.set_ticks_position('top')
cb2.set_label('CH$_4$ mole fraction (ppb)')
#cb2_ticks = [obs_min, (obs_min+obs_max)//2,obs_max]
#cb2.set_ticks([cb2_ticks])

# Draw lines between 2 maps to show zoom box
xyA0 = (map_extents0[0],map_extents0[2])
xyB0 = (map_extents[0],map_extents[2])
xyB1 = (map_extents[0],map_extents[3])

con = mpatches.ConnectionPatch(xyA = xyB0, xyB=xyB0, coordsA="data", 
                               coordsB="data", axesA = ax1, axesB=ax0, color="gold" )
ax1.add_artist(con)

con2 = mpatches.ConnectionPatch(xyA = xyB1, xyB=xyB1, coordsA="data", 
                               coordsB="data", axesA = ax1, axesB=ax0, color="gold" )
ax1.add_artist(con2)

bounds4 = np.linspace(-32, 32, 17)
norm4 = colors.BoundaryNorm(boundaries=bounds4, ncolors=256)

h4 = ax4.pcolormesh(lon_map-dlon/2, lat_map-dlat/2, map_data, transform=ccrs.PlateCarree(), cmap='RdBu_r', norm=norm4)
ax4.coastlines()
ax4.add_feature(cfeature.BORDERS)

#h3 = ax3.pcolormesh(lon_map-dlon/2, lat_map-dlat/2, map_data, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-35, vmax=35)
#ax3.coastlines()
#ax3.add_feature(cfeature.BORDERS)
#cbaxes4 = fig2.add_axes([0.89, 0.65, 0.015, 0.23]) 

cbaxes4 = fig2.add_axes([0.6, 0.18, 0.23, 0.02]) 
cb4=plt.colorbar(h4,cax=cbaxes4,orientation = "horizontal", extend='both')
#cb4.ax.xaxis.set_ticks_position('top')
cb4.set_label("GOSAT XCH$_4$ - model XCH$_4$ (ppb)")
#cb4_ticks = [0, 6,12,18,24]
cb4_ticks = [-32, -16,0, 16,32]
#cb4_ticks = [0,8,16,24,32,40]
cb4.set_ticks([cb4_ticks])

ax4.set_extent(map_extents0, crs=ccrs.PlateCarree())

gl4 = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl4.xlabels_top = False
gl4.ylabels_left = False
gl4.xlocator = mticker.FixedLocator([-85,-80,-75,-70, -65, -60, -55, -45])
gl4.ylocator = mticker.FixedLocator([-28,-24, -20,-16,-12, -8,-4,0,8])#            "20180701", "20180716",
gl4.xformatter = LONGITUDE_FORMATTER
gl4.yformatter = LATITUDE_FORMATTER

n = 3
pcolors = plt.cm.viridis(np.linspace(0,1,n))
mksize=3

ax2.plot(y_obsA.time,y_obsA, 'o', markersize=mksize, color="black", linewidth=2, label="Observations")
ax2.plot(y_time_gcA,y_post_gcA, 's', markersize=mksize, color=pcolors[1], linewidth=2, label="GEOS-Chem posterior")
ax2.plot(y_time_gcA,y_post_nameA, '^', markersize=mksize, color="C4", linewidth=2, label="NAME posterior")

ax2.plot(y_time_gcA,y_ap_gcA, linestyle='--', color="grey", label="GEOS-Chem prior")
ax2.plot(y_time_gcA,y_ap_nameA, linestyle=':', markersize=mksize, color="grey", label="NAME prior")
#ax.plot(y_post2, color="C3", linestyle="--", label="Posterior model (WetCHARTs)")
#ax.plot(H[:,0]*x_post[0], color="C1", linestyle=':', label="Posterior baseline")

ax3.plot(y_obsB.time,y_obsB, 'o', markersize=mksize, color="black", linewidth=2, label="Observations")
ax3.plot(y_time_gcB,y_post_gcB, 's', markersize=mksize, color=pcolors[1], linewidth=2, label="GEOS-Chem posterior")
ax3.plot(y_time_gcB,y_post_nameB, '^', markersize=mksize, color="C4", linewidth=2, label="NAME posterior")

ax3.plot(y_time_gcB,y_ap_gcB, linestyle='--', color="grey", label="GEOS-Chem prior")
ax3.plot(y_time_gcB,y_ap_nameB, linestyle=':', markersize=mksize, color="grey", label="NAME prior")

ax2.set_ylabel("CH$_4$ (ppb)")
ax3.set_ylabel("CH$_4$ (ppb)")

ax3.set_xlabel("Time (UTC)")
ax2.legend(ncol=2, bbox_to_anchor=(1.04,1), loc="upper left")

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

txt0 = ax0.text(0.05,0.1, "A", transform=ax0.transAxes, fontsize=14)
txt1 = ax1.text(0.05,0.1, "B", transform=ax1.transAxes, fontsize=14)
txt4 = ax4.text(0.05,0.1, "E", transform=ax4.transAxes, fontsize=14, color="black")
txt2 = ax2.text(0.02,0.05, "C", transform=ax2.transAxes, fontsize=14)
txt3 = ax3.text(0.02,0.05, "D", transform=ax3.transAxes, fontsize=14)

ax0.set_aspect('auto')
ax1.set_aspect('auto')
