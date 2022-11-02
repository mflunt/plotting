#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:55:30 2022

@author: mlunt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:39:51 2021

Read in Hydro Basins shapefiles 

Need to find a naming convention so I know what basin is what.
Maybe use PFAF_ID and use first 2 digits as way of identifying NILE and EAF?

@author: mlunt
"""

import geopandas
from rasterio import features
from affine import Affine
import xarray
import numpy as np
import regionmask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.
    """
    transform = transform_from_latlon(coords['lat'], coords['lon'])
    out_shape = (len(coords['lat']), len(coords['lon']))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    return xarray.DataArray(raster, coords=coords, dims=('lat', 'lon'))

lev = "02"

#latmin = -36
#latmax = 30
#lonmin = -20   # was 0
#lonmax = 55

latmin = -90
latmax = 90
lonmin = -180   # was 0
lonmax = 180

dlon=0.625
dlat=0.5

# Set up mask dataset for 0.25x0.3125 grid
ds = xarray.Dataset(coords={'lon': np.arange(lonmin,lonmax, dlon),
                          'lat': np.arange(latmin,latmax,dlat)})

# this shapefile is from natural earth data
# http://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/
states_af = geopandas.read_file('/home/mlunt/datastore/MOYA/africa/HydroBASINS/hybas_lake_af_lev' + lev + '_v1c.shp')
states_nam = geopandas.read_file('/home/mlunt/datastore/MOYA/africa/HydroBASINS/NAM/hybas_na_lev' + lev + '_v1c.shp')
states = geopandas.read_file('/home/mlunt/datastore/MOYA/africa/HydroBASINS/SAM/hybas_lake_sa_lev' + lev + '_v1c.shp')

#states = geopandas.read_file('/home/mlunt/datastore/TRMM/Lake_Victoria_Basin_Shapefile/LV_Basin.shp')

ID = states["HYBAS_ID"]
ID_af = states_af["HYBAS_ID"]
ID_nam = states_nam["HYBAS_ID"]

lon = np.arange(lonmin,lonmax+dlon,dlon)
lat = np.arange(latmin,latmax+dlat,dlat)

#mask = regionmask.mask_geopandas(states,lon,lat)

nbase = len(ID)
nbase_af = len(ID_af)
nbase_nam = len(ID_nam)

count=1
count_af=1
count_nam=1

for xi in range(nbase):

    us_states = states.iloc[xi]
    
    if type(states.iloc[xi].geometry) == type(states.iloc[0].geometry):
        shapes = [(shape, n) for n, shape in enumerate(us_states.geometry)]
    else:
        shapes = [(us_states.geometry,0)]

    ds["SA_basin_" + str(count).zfill(2)] = rasterize(shapes, ds.coords)
    count = count + 1
    
    
for xi in range(nbase_af):

    af_states = states_af.iloc[xi]
    
    if type(states_af.iloc[xi].geometry) == type(states_af.iloc[0].geometry):
        shapes = [(shape, n) for n, shape in enumerate(af_states.geometry)]
    else:
        shapes = [(af_states.geometry,0)]

    ds["AF_basin_" + str(count_af).zfill(2)] = rasterize(shapes, ds.coords)
    count_af = count_af + 1
    
for xi in range(nbase_nam):

    nam_states = states_nam.iloc[xi]
    
    if type(states_nam.iloc[xi].geometry) == type(states_nam.iloc[0].geometry):
        shapes = [(shape, n) for n, shape in enumerate(nam_states.geometry)]
    else:
        shapes = [(nam_states.geometry,0)]

    ds["NA_basin_" + str(count_nam).zfill(2)] = rasterize(shapes, ds.coords)
    count_nam = count_nam + 1
    

#%%

ds2=ds.copy()
for ti in range(1, count):
    dum = ds["SA_basin_" + str(ti).zfill(2)].copy()
    if dum.max() >0:
        new_da = np.ceil(dum/dum.max())
    else:
        new_da = dum +1.
        
    ds2["SA_basin_" + str(ti).zfill(2)] = new_da
    
for ti in range(1, count_af):
    dum = ds["AF_basin_" + str(ti).zfill(2)].copy()
    if dum.max() >0:
        new_da = np.ceil(dum/dum.max())
    else:
        new_da = dum +1.
        
    ds2["AF_basin_" + str(ti).zfill(2)] = new_da
    
for ti in range(1, count_nam):
    dum = ds["NA_basin_" + str(ti).zfill(2)].copy()
    if dum.max() >0:
        new_da = np.ceil(dum/dum.max())
    else:
        new_da = dum +1.
        
    ds2["NA_basin_" + str(ti).zfill(2)] = new_da

    
#da = ds2.basin_01.fillna(0.)
#for ti in range(2, count):
#    da = da + ds2["basin_" + str(ti).zfill(2)].fillna(0.)
#    
#    
#if lev == "02":
#    basin_names = {"basin_01": "E_Africa", 
#                   "basin_02": "S_Africa", 
#                   "basin_03": "Congo", 
#                   "basin_04": "W_Africa", 
#                   "basin_05": "Sahara", 
#                   "basin_06": "Nile", 
#                   "basin_07": "Madagascar", 
#                   "basin_08": "Chad",   
#                   }
#    
#    ds_out = ds2.rename(name_dict = basin_names)
#    
#ds3=xarray.Dataset()
#if lev =="03":
#    # Loop through basins if no finite indices then drop
#    ds_keys = list(ds2.keys())
#    basin_count=1
#    for key in ds_keys:
#        if ds2[key].sum() >350:
#            print(ds2[key].sum())
#            ds3["basin_" + str(basin_count).zfill(2)] = ds2[key]
#            basin_count+=1
#            
#    basin_names = {"basin_01": "Horn", 
#                   "basin_02": "Jubba", 
#                   "basin_03": "Tana", 
#                   "basin_04": "Congo", 
#                   "basin_05": "Nile", 
#                   "basin_06": "Rift", 
#                   }
#    ds_out = ds3.rename(name_dict = basin_names) 
#    ds_out.coords["lat"] = lat
#    ds_out.coords["lon"] = lon
#
#if lev =="04":
#    # Loop through basins if no finite indices then drop
#    ds_keys = list(ds2.keys())
#    basin_count=1
#    for key in ds_keys:
#        if ds2[key].sum() >200:
#            print(ds2[key].sum())
#            ds3["basin_" + str(basin_count).zfill(2)] = ds2[key]
#            basin_count+=1
#            

#%%
count2 = 1
count_na2 = 1
count_af2 = 1
da_basin = ds2["SA_basin_01"].fillna(0.)*0.
for ti in range(1, count):
    da_basin = da_basin + ds2["SA_basin_" + str(ti).zfill(2)].fillna(0.)*count2
    count2+=1
    
for ti in range(1, count_nam):
    da_basin = da_basin + ds2["NA_basin_" + str(ti).zfill(2)].fillna(0.)*count_na2
    count_na2+=1
    
for ti in range(1, count_af):
    da_basin = da_basin + ds2["AF_basin_" + str(ti).zfill(2)].fillna(0.)*count_af2
    count_af2+=1
    
da_basin2 = da_basin.where(da_basin >0)
proj = ccrs.PlateCarree()
fig,ax=plt.subplots(subplot_kw=dict(projection=proj))
h3 = ax.pcolormesh(lon, lat,da_basin2, transform=ccrs.PlateCarree(), cmap='viridis', vmin=1, vmax=8)
ax.coastlines()
    
   
    
    
#proj = ccrs.PlateCarree()
#fig,axes=plt.subplots(2,4,subplot_kw=dict(projection=proj))
#axs=axes.ravel()
#for ti in range(7):
#    h3 = axs[ti].pcolormesh(lon, lat,ds2["basin_" + str(ti+1).zfill(2)], transform=ccrs.PlateCarree(), cmap='viridis', vmin=0.5, vmax=1.5)
#    axs[ti].coastlines()
#    axs[ti].add_feature(cfeature.BORDERS, edgecolor='black', alpha=0.7)
#    axs[ti].set_title(str(ti+1))

# Save mask file to disk

    
out_dir = '/home/mlunt/datastore/MOYA/africa/HydroBASINS/SAM/'
#fname_out = out_dir + "Africa_level2_basins_mask.nc"
fname_out = out_dir + "AF_NA_SA_level2_basins_mask_05x0625.nc"

#ds2.to_netcdf(path=fname_out)
    
