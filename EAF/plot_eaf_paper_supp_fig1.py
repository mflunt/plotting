#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:55:13 2022

Plot Supplementary figure for EAF paper showing:
    
a) Basis function grid
    
b) Prior emissions mean
    
c) 1 iteration of posterior scale factors
    
d) Posterior scale factor mean from one month (or one year))

@author: mlunt
"""

import xarray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colorbar as cbar

def open_ds(fname, group=None):
    with xarray.open_dataset(fname) as ds:
        ds.load()
    return ds

#%%
# Read in inner and outer basis function files
basis_dir = "/home/mlunt/ceph/verify/model_settings/DARE_runs/CH4/eaf_runs/masks/"

nbasis_inner = 120
nbasis_outer = 98

fname_inner = basis_dir + "eaf_bfs_inner.nc"
fname_outer = basis_dir + "eaf_bfs_outer.nc"

ds_inner = open_ds(fname_inner)
ds_outer = open_ds(fname_outer)

lon = ds_inner.lon.values
lat = ds_inner.lat.values

dlon = lon[1]-lon[0]
dlat = lat[1]-lat[0]

keys_inner = []
keys_outer = []

for xi in range(nbasis_inner):
    keys_inner.append("bf_" + str(xi+1).zfill(2))
    if xi < nbasis_outer:
        keys_outer.append("bf_" + str(xi+1).zfill(2))

# loop through keys and create ascending basis function map
bf_map2 = ds_inner.bf_01.squeeze()*0.

for xi,reg in enumerate(keys_inner):
    bf_map2 = bf_map2 + ds_inner[reg].squeeze()*(xi+1)

for xi,reg in enumerate(keys_outer):
    bf_map2 = bf_map2 + ds_outer[reg].squeeze()*(xi+1+nbasis_inner)

bf_map = bf_map2.where(bf_map2>0)

#%%
# Read in ppost file containing prior emissions
post_dir = "/home/mlunt/datastore/EAF/inv_outputs/paper/emissions/" 
fname_inv = post_dir +  "post_emissions_gosat_ensemb10.nc"

ds_inv = open_ds(fname_inv)

emis_ap = ds_inv.emis_ens_ap.mean(dim=["ens", "time"])


#%%
# Read in individual monthly output file
it_dir = "/home/mlunt/datastore/EAF/inv_outputs/paper/eaf_transd_gs_100cm_lst0/" 
fname_it = it_dir + "inv_out_eaf_transd_gs_100cm_lst0_20170401.nc"

ds_it = open_ds(fname_it)

inner_it = ds_it.x_it_inner   # (nbasis_inner, nIt)
outer_it = ds_it.x_it_outer

it = 204   # 200 has 49 unknowns

inner_mean = inner_it.mean(dim="nIt")
outer_mean = outer_it.mean(dim="nIt")

x_post_it = ds_inner["bf_01"].squeeze()*0.
x_post_mean = ds_inner["bf_01"].squeeze()*0.

for xi, reg in enumerate(keys_inner):
    x_post_it = x_post_it + ds_inner[reg].squeeze()*inner_it[xi,it]
    x_post_mean = x_post_mean + ds_inner[reg].squeeze()*inner_mean[xi]

for xi, reg in enumerate(keys_outer):
    x_post_it = x_post_it + ds_outer[reg].squeeze()*outer_it[xi,it]
    x_post_mean = x_post_mean + ds_outer[reg].squeeze()*outer_mean[xi]


#%%
# Plot outputs in a 4-panel figure
alphabet = ["A", "B", "C", "D"]
proj = ccrs.PlateCarree()
fig,axes = plt.subplots(2,2, subplot_kw=dict(projection=proj),figsize=(8,8))

axs=axes.ravel()
for ti in range(4):
    axs[ti].coastlines()
    axs[ti].add_feature(cfeature.BORDERS)
    
# Plot 1 - basis function map
h0 = axs[0].pcolormesh(lon-dlon/2., lat-dlat/2., bf_map, transform=ccrs.PlateCarree(), 
        cmap='Spectral', vmin=0, vmax=nbasis_inner + nbasis_outer)

# Plot 2 - prior emissions distribution
h1 = axs[1].pcolormesh(lon-dlon/2., lat-dlat/2., emis_ap, transform=ccrs.PlateCarree(), 
        cmap='viridis', vmin=0, vmax=5e-10)

# Plot 3 - single iteration of scale factors
h2 = axs[2].pcolormesh(lon-dlon/2., lat-dlat/2., np.log2(x_post_it), transform=ccrs.PlateCarree(), 
        cmap='RdBu_r', vmin=-2, vmax=2)

# Plot 4 - scale factors across all iterations
h3 = axs[3].pcolormesh(lon-dlon/2., lat-dlat/2., np.log2(x_post_mean), transform=ccrs.PlateCarree(), 
        cmap='RdBu_r', vmin=-2, vmax=2)

##[left, bottom, width, height],
cbaxes0 = fig.add_axes([0.13, 0.52, 0.35, 0.02]) 
cb0 = plt.colorbar(h0, cax = cbaxes0, orientation='horizontal', extend='neither', label = 'Basis function count []')

cbaxes1 = fig.add_axes([0.55, 0.52, 0.35, 0.02]) 
cb1 = plt.colorbar(h1, cax = cbaxes1, orientation='horizontal', extend='max', label = 'CH$_4$ emissions [kg m$^{-2}$ s$^{-1}$]')

cbaxes2 = fig.add_axes([0.13, 0.1, 0.35, 0.02]) 
cb2 = plt.colorbar(h2, cax = cbaxes2, orientation='horizontal', extend='both', label = 'log$_2$ posterior scale factor []')

cbaxes3 = fig.add_axes([0.55, 0.1, 0.35, 0.02]) 
cb3 = plt.colorbar(h3, cax = cbaxes3, orientation='horizontal', extend='both', label = 'log$_2$ posterior scale factor []')

for ti in range(4):
    axs[ti].text(-0.1,0.85, alphabet[ti], transform=axs[ti].transAxes, fontsize=11)

fname_fig = "/home/mlunt/results/EAF_project/paper/supp_fig_v1.png"
plt.savefig(fname_fig)