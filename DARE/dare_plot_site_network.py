#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:22:26 2022

Plot ICOS and DECC sites on map

@author: mlunt
"""

import xarray
import json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import numpy as np

def open_ds(fname):
    with xarray.open_dataset(fname) as ds:
        ds.load()
    return ds

co2_sites = ["GAT",  "HTM", 
        "KIT", "KRE", "LIN", "NOR", "OPE", 
        "SAC" , "SVB", "TOH", "TRN", 
        "RGL", "BSD", "HFD", "TAC",
        "CBW",  "MHD",   # "CRP", "MLH",
        "HUN", "BRM", "OHP"]

co_sites = ["GAT", "HTM",  
        "KRE",  "NOR", "OPE",
        "SAC" ,  "SVB", "TRN", 
        "TAC",  "BSD", "HFD"]

# Get site info data
json_dir = "/home/mlunt/programs/Python/DARE/inv_code/"
# Read in json file
with open(json_dir + "dare_site_info.json") as f:
    site_info=json.load(f)

# Read in TNO CO2 data
tno_dir = "/home/mlunt/ceph/verify/model_settings/DARE_runs/inv_runs/emissions/processed/"
fname_tno = tno_dir + "processed_TNO_co2ff_clim.nc"
ds_tno = open_ds(fname_tno)

fname_tno_co = tno_dir + "processed_TNO_coff_clim.nc"
ds_tno_co = open_ds(fname_tno_co)

emis_tno = ds_tno.co2ff.mean(dim="time")
lon = emis_tno.lon.values
lat  = emis_tno.lat.values
dlon = lon[1]-lon[0]
dlat = lat[1]-lat[0]

emis_tno_co = ds_tno_co.coff.mean(dim="time")

#%%
# Read in basis function file

basis_dir = "/home/mlunt/ceph/verify/model_settings/DARE_runs/inv_runs/masks/"
fname_basis = basis_dir + "DARE_basis_functions_0.25x0.3125.nc"
ds_basis = open_ds(fname_basis)

basis_keys = []
nbasis=109
for xi in range(1, nbasis+1):
    basis_keys.append("bf_"+ str(xi).zfill(2))

bfs = ds_basis["bf_01"].squeeze()*0.

for xi, reg in enumerate(basis_keys):

    bfs = bfs + ds_basis[reg].squeeze() * (xi+1)


#%%
alphabet = ["(a)", "(b)", "(c)", "(d)"  ]

cmin = 0.
cmax = 3.e-7

cmin2 = 0.
cmax2 = 1.e-9

proj = ccrs.PlateCarree()
fig,axes = plt.subplots(2,2,subplot_kw=dict(projection=proj),figsize=(8,6))

axs = axes.ravel()

for ax in axs:
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent((-14,25, 40,64))

# Subplot 1 - sites

lat_co2_si=[]
lon_co2_si=[]
lat_coco2_si=[]
lon_coco2_si=[]
for site in co2_sites:
    if site in co_sites:
        pass
    else:
        dict_si = site_info[site]
        network_si = dict_si["network"]
        
        lat_si = dict_si["latitude"]
        lon_si = dict_si["longitude"]
        
        lat_co2_si.append(lat_si)
        lon_co2_si.append(lon_si)
        
        #ax.scatter(lon_si,lat_si, color='C6', s=16)
        #axs[0].text(lon_si+0.2,lat_si+0.2, site, fontsize=11)
    
for site in co_sites:
    dict_si = site_info[site]
    network_si = dict_si["network"]
    
    lat_si = dict_si["latitude"]
    lon_si = dict_si["longitude"]
    
    lat_coco2_si.append(lat_si)
    lon_coco2_si.append(lon_si)
        
    #ax.scatter(lon_si,lat_si, color='C9', s=16, label = "CO and CO$_2$")
    #axs[0].text(lon_si+0.2,lat_si+0.2, site, color='C9', fontsize=11)


axs[0].scatter(lon_co2_si,lat_co2_si, color='C6', s=16, label  = "CO$_2$")
axs[0].scatter(lon_coco2_si,lat_coco2_si, color='C9', s=16, label = "CO and CO$_2$")
 
axs[0].legend(fontsize=11, ncol=2, bbox_to_anchor=(0.58, -0.1, 0.4, 0.1))


# Subplot 2 - Basis functions

p1 = axs[1].pcolormesh(lon-dlon/2., lat-dlat/2., bfs,
 transform=ccrs.PlateCarree(), cmap='Spectral', vmin=0, vmax=109)

cbaxes1 = fig.add_axes([0.545, 0.52, 0.35, 0.02]) 
cb2 = plt.colorbar(p1, cax = cbaxes1, orientation='horizontal', extend='neither')
cb2.set_label('Basis function count []', size=11)

# Subplot 3 - CO2 emissions

h2a1 = axs[2].pcolormesh(lon-dlon/2., lat-dlat/2., emis_tno,
 transform=ccrs.PlateCarree(), cmap='OrRd', vmin=cmin, vmax=cmax)

#cbaxes2 = fig2.add_axes([0.82, 0.12, 0.02, 0.76]) 
cbaxes2 = fig.add_axes([0.125, 0.09, 0.35, 0.02]) 
cb2 = plt.colorbar(h2a1, cax = cbaxes2, orientation='horizontal', extend='max')
cb2.set_label('CO$_{2}^{ff}$ emissions (kg m$^{-2}$ s$^{-1}$)', size=11)

# Subplot 4 - CO emissions

h2a2 = axs[3].pcolormesh(lon-dlon/2., lat-dlat/2., emis_tno_co,
 transform=ccrs.PlateCarree(), cmap='OrRd', vmin=cmin2, vmax=cmax2)

#cbaxes2 = fig2.add_axes([0.82, 0.12, 0.02, 0.76]) 
cbaxes3 = fig.add_axes([0.545, 0.09, 0.35, 0.02]) 
cb2 = plt.colorbar(h2a2, cax = cbaxes3, orientation='horizontal', extend='max')
cb2.set_label('CO$^{ff}$ emissions (kg m$^{-2}$ s$^{-1}$)', size=11)

for xi,ax in enumerate(axs):

    ax.text(0.07,0.85, alphabet[xi], transform = ax.transAxes, fontsize=11)

fname_out = "/home/mlunt/results/DARE/paper/figs_draft_202209/sites_emis_bf.pdf"
plt.savefig(fname_out)