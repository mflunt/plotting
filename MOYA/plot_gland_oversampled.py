#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 12:44:34 2021

Plot oversampled TROPOMI 

@author: mlunt
"""
import numpy as np
import xarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colorbar as cbar
from cartopy.io.img_tiles import GoogleTiles as OSM

import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from enkf_code import scalebar_mod

def open_ds(fname):
    with xarray.open_dataset(fname) as ds:
        ds.load()
    return ds

year = "2019"

lonmin = -43.
lonmax = -37.
latmin = 64.
latmax= 67.


#fname_trop = "/home/mlunt/datastore/MOYA/oz/TROPOMI_002x002_5km_Bowen_" + year + "_easterly.nc"
fname_trop = "/home/mlunt/datastore/MOYA/greenland/TROPOMI_002x002_5km_GLAND_2019.nc"
#fname_trop = "/home/mlunt/datastore/MOYA/greenland/TROPOMI_002x002_5km_GLAND_unfiltered_2019.nc"
#fname_trop = "/home/mlunt/datastore/MOYA/greenland/TROPOMI_005x005_10km_GLAND_unfiltered_2019.nc"
ds_trop = open_ds(fname_trop)
ds_trop2 = ds_trop.sel(lon=slice(lonmin,lonmax), lat=slice(latmin,latmax))
lon = ds_trop2.lon
lat = ds_trop2.lat
ch4 = ds_trop2.ch4

count = ds_trop2["count"]
#tiler = OSM(style='terrain-background')
tiler = OSM(style='satellite')
mercator = tiler.crs

#tiler2 = OSM(style='street')
#mercator2 = tiler2.crs

#tiler = GoogleTiles()
#mercator = tiler.crs
#ax = plt.axes(projection=mercator)

#%%
fig,ax=plt.subplots(subplot_kw=dict(projection=mercator), figsize=(10,10))

if year == "2019":
    cmax=1880
    cmin=1820
elif year == "2018":
    cmax=1870
    cmin=1835

bounds = np.linspace(cmin, cmax, 11)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

lon2 = lon-0.02/2.
lat2 = lat-0.02/2.

ax.set_extent((lonmin, lonmax, latmin, latmax))
zoom = 9  #6 or 9 depending on zoom
ax.add_image(tiler, zoom )

#scatter_map = 'viridis'
#h3 = ax.pcolormesh(lon, lat, ch4,
#                transform=ccrs.PlateCarree(), cmap='viridis', alpha=0.35, vmin = 1815., vmax=1825 )
#ch4_nan = ch4.copy()
ch4_nan = ch4.where(ch4 > 1800)

h3 = ax.pcolormesh(lon, lat, ch4_nan,
                transform=ccrs.PlateCarree(), cmap='PuRd', alpha=0.6, norm=norm )
#h3 = ax.contourf(lon, lat, ch4, levels=bounds,
#                transform=ccrs.PlateCarree(), cmap='viridis', alpha=0.4, extend="both")

#cbaxes = fig.add_axes([0.1, 0.06, 0.8, 0.02])
cbaxes = fig.add_axes([0.90, 0.1, 0.02, 0.8]) 
##[left, bottom, width, height],
cb = plt.colorbar(h3, cax = cbaxes, orientation='vertical', extend='both', label = 'CH4 mole fraction (ppb)') 

#scalebar_mod.scale_bar(ax, (0.15, 0.2), 20, text_kwargs = {"fontsize": 12})

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = True
gl.ylabels_right = False
#gl.xlines = True
gl.xlocator = mticker.FixedLocator([-45, -44, -43, -42,-41, -40,-39, -38,-36,])
gl.ylocator = mticker.FixedLocator([63,64,65,66,67,68])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
#gl.xlabel_style = {'size': 15, 'color': 'gray'}
#gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

#%%
"""

fig,ax=plt.subplots(subplot_kw=dict(projection=mercator), figsize=(10,10))

lon2 = lon-0.02/2.
lat2 = lat-0.02/2.

#rivers_50m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '50m')

ax.set_extent((lonmin, lonmax, latmin, latmax))
zoom = 12
ax.add_image(tiler, zoom )

h3 = ax.pcolormesh(lon, lat, count,
                transform=ccrs.PlateCarree(), cmap='viridis', alpha=0.35, vmin = 0., vmax=50. )

cbaxes = fig.add_axes([0.1, 0.06, 0.8, 0.02]) 
##[left, bottom, width, height],
cb = plt.colorbar(h3, cax = cbaxes, orientation='horizontal', extend='both', label = 'Obs count') 

ax.add_patch(mpatches.Rectangle(xy=[mor_lons[0], mor_lats[0]], 
                                 width=mor_lons[1]-mor_lons[0], 
                                 height=mor_lats[1] - mor_lats[0],
                                    fill=False, edgecolor="black",
                                    alpha=1,
                                    transform=ccrs.PlateCarree()))

ax.add_patch(mpatches.Rectangle(xy=[sw_lons[0], sw_lats[0]], 
                                 width=sw_lons[1]-sw_lons[0], 
                                 height=sw_lats[1] - sw_lats[0],
                                    fill=False, edgecolor="black",
                                    alpha=1,
                                    transform=ccrs.PlateCarree()))

ax.add_patch(mpatches.Rectangle(xy=[hc_lons[0], hc_lats[0]], 
                                 width=hc_lons[1]-hc_lons[0], 
                                 height=hc_lats[1] - hc_lats[0],
                                    fill=False, edgecolor="black",
                                    alpha=1,
                                    transform=ccrs.PlateCarree()))

ax.add_patch(mpatches.Rectangle(xy=[gt_lons[0], gt_lats[0]], 
                                 width=gt_lons[1]-gt_lons[0], 
                                 height=gt_lats[1] - gt_lats[0],
                                    fill=False, edgecolor="black",
                                    alpha=1,
                                    transform=ccrs.PlateCarree()))
"""