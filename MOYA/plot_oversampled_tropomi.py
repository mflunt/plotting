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

gwp = 28.   # Use AR5 100-year value as AUstralia NGER now does. 

year = "2019"
mines = ["Moranbah", "Coppabella", "Hail_creek", "Capcoal"]
mines_short={"Moranbah": "MN",
            "Coppabella": "CB",
            "Hail_creek": "HC",
            "Capcoal": "CC"}
lon_mines={"Moranbah": 147.96,
    "Coppabella": 148.44,
        "Hail_creek": 148.36,
            "Capcoal": 148.58}
    
lat_mines={"Moranbah": -21.88,
    "Coppabella": -21.85,
        "Hail_creek": -21.46,
            "Capcoal": -22.99}

# Moranbah North
#site_lons={"Moranbah": [147.90,148.00],    #  [147.86,148.04],
#            "Coppabella":[148.42, 148.48],  #  [148.42,148.48],
#            "Hail_creek":[148.36, 148.40],  # [148.34, 148.44],
#            "Capcoal": [148.54, 148.64] }  # [148.50, 148.64]

#site_lats={"Moranbah": [-21.90, -21.82],    # [-21.94, -21.78]
#            "Coppabella":[-21.88, -21.84],  # [-21.88, -21.82],
#            "Hail_creek": [-21.50, -21.44],  #  [-21.50, -21.40],
#            "Capcoal": [-23.02, -22.94] }  # [-23.04, -22.92]

site_lons={"Moranbah": [147.80,148.00],    #  [147.86,148.04], # Try Moranbah with Goonyella
            "Coppabella":[148.36, 148.48],  #  [148.42,148.48], # Hail Creek and Coppabella are challenging.
            "Hail_creek":[148.32, 148.42],  # [148.34, 148.44],
            "Capcoal": [148.46, 148.64] }  # [148.50, 148.64]

site_lats={"Moranbah": [-21.96, -21.76],    # [-21.94, -21.78]
            "Coppabella":[-21.88, -21.76],  # [-21.88, -21.82],
            "Hail_creek": [-21.50, -21.40],  #  [-21.50, -21.40],
            "Capcoal": [-23.08, -22.90] }  # [-23.04, -22.92]

bg_site_lons = {"Moranbah": [148.62,148.72],
            "Coppabella":[148.62, 148.72],
            "Hail_creek":[148.62, 148.72],
            "Capcoal": [148.94, 149.04] }   

bg_site_lats={"Moranbah": [-21.80, -21.70],
            "Coppabella":[-21.80, -21.70],
            "Hail_creek": [-21.80, -21.70],
            "Capcoal": [-22.88, -22.78] }




#lonmin = 147.
#lonmax = 150.
#latmin = -24.
#latmax= -20.5

# Hunter Valley 
#lonmin = 149.5
#lonmax = 152.5
#latmin = -34.5
#latmax= -31.5

# Near Moranbah North / Goonyella
#lonmin = 147.5
#lonmax = 148.8
#latmin = -22.4
#latmax = -21.25

# Near Moranbah North / Goonyella
lonmin = 147.5
lonmax = 149.1
latmin = -23.6
latmax = -21.1

lonmin2 = 147.5
lonmax2 = 148.9
latmin2 = -23.3
latmax2 = -21.28

# Wider area
#lonmin = 145.2
#lonmax = 149.8
#latmin = -24.8
#latmax = -20.1
#
#lonmin2 = 145.3
#lonmax2 = 149.8
#latmin2 = -24.8
#latmax2 = -20.1


# Near Grasstree mine
#lonmin = 147.9
#lonmax = 149.1
#latmin = -23.6
#latmax = -22.4
fname_trop = "/home/mlunt/datastore/MOYA/oz/TROPOMI_002x002_5km_Bowen_" + year + "_easterly.nc"
#fname_trop = "/home/mlunt/datastore/MOYA/oz/TROPOMI_005x005_5km_Bowen_" + year + "_easterly.nc"
ds_trop = open_ds(fname_trop)
ds_trop2 = ds_trop.sel(lon=slice(lonmin,lonmax), lat=slice(latmin,latmax))
lon = ds_trop2.lon
lat = ds_trop2.lat
ch4 = ds_trop2.ch4

count = ds_trop2["count"]
#tiler = OSM(style='terrain-background')
tiler = OSM(style='satellite')
mercator = tiler.crs

tiler2 = OSM(style='street')
mercator2 = tiler2.crs

#tiler = GoogleTiles()
#mercator = tiler.crs
#ax = plt.axes(projection=mercator)

#%%
fig,axes=plt.subplots(1,2,subplot_kw=dict(projection=mercator), figsize=(12,10))

if year == "2019":
    cmax=1828
    #cmin=1810
    cmin=1808
    #cmax=1824
    #cmin=1808
elif year == "2018":
    cmax=1820
    cmin=1804

bounds = np.linspace(cmin, cmax, 11)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

ax0 = axes[0]
ax = axes[1]
#fig=plt.figure(figsize=(9,10))

map_extents0 = [140, 155., -36., -16.]
ax0.set_extent(map_extents0, crs=ccrs.PlateCarree())

#tiler2 = OSM(style='terrain')
#mercator2 = tiler2.crs
ax0.add_image(tiler2,5)
ax0.add_patch(mpatches.Rectangle(xy=[lonmin2, latmin2], 
                                 width=lonmax2-lonmin2, 
                                 height=latmax2-latmin2,
                                    fill=False, edgecolor="black",
                                    alpha=1,
                                    transform=ccrs.PlateCarree()))


lon2 = lon-0.02/2.
lat2 = lat-0.02/2.



ax.set_extent((lonmin2, lonmax2, latmin2, latmax2))
zoom = 12
ax.add_image(tiler, zoom )

#scatter_map = 'viridis'
#h3 = ax.pcolormesh(lon, lat, ch4,
#                transform=ccrs.PlateCarree(), cmap='viridis', alpha=0.35, vmin = 1815., vmax=1825 )
#ch4_nan = ch4.copy()
ch4_nan = ch4.where(ch4 >= 1810)

h3 = ax.pcolormesh(lon, lat, ch4,
                transform=ccrs.PlateCarree(), cmap='RdYlBu_r', alpha=0.4, norm=norm )
#h3 = ax.contourf(lon, lat, ch4, levels=bounds,
#                transform=ccrs.PlateCarree(), cmap='viridis', alpha=0.4, extend="both")

#cbaxes = fig.add_axes([0.1, 0.06, 0.8, 0.02])
cbaxes = fig.add_axes([0.91, 0.14, 0.02, 0.71]) 
##[left, bottom, width, height],
cb = plt.colorbar(h3, cax = cbaxes, orientation='vertical', extend='both', label = 'XCH$_4$ mole fraction (ppb)') 
#cb = plt.colorbar(h3, cax = cbaxes, orientation='vertical', extend='max', label = 'XCH$_4$ mole fraction (ppb)') 

scalebar_mod.scale_bar(ax, (0.15, 0.2), 20, text_kwargs = {"fontsize": 14})

# Draw lines between 2 maps to show zoom box

#xyB0 = (lonmin2+0.1, latmin2+0.1)
#xyB1 = (lonmin2+0.1, latmax2-0.1)

use_proj = mercator
use_proj2 = mercator2
xyA0 = use_proj.transform_point(lonmin2, latmin2, ccrs.PlateCarree())
xyB0 = use_proj2.transform_point(lonmin2, latmin2, ccrs.PlateCarree())

xyA1 = use_proj.transform_point(lonmin2, latmax2, ccrs.PlateCarree())
xyB1 = use_proj2.transform_point(lonmin2, latmax2, ccrs.PlateCarree())

con = mpatches.ConnectionPatch(xyA = xyA0, xyB=xyB0, coordsA="data", 
                               coordsB="data", axesA = axes[1], axesB=axes[0])
axes[1].add_artist(con)

con2 = mpatches.ConnectionPatch(xyA = xyA1, xyB=xyB1, coordsA="data", 
                               coordsB="data", axesA = axes[1], axesB=axes[0] )
axes[1].add_artist(con2)

for mine in mines:
    lons_si = site_lons[mine]
    lats_si = site_lats[mine]

    bg_lons_si = bg_site_lons[mine]
    bg_lats_si = bg_site_lats[mine]

    

    ax.add_patch(mpatches.Rectangle(xy=[lons_si[0], lats_si[0]], 
                                 width=lons_si[1]-lons_si[0], 
                                 height=lats_si[1] - lats_si[0],
                                    fill=False, edgecolor="black",
                                    alpha=1,
                                    transform=ccrs.PlateCarree()))

#    ax.add_patch(mpatches.Rectangle(xy=[bg_lons_si[0], bg_lats_si[0]], 
#                                 width=bg_lons_si[1]-bg_lons_si[0], 
#                                 height=bg_lats_si[1] - bg_lats_si[0],
#                                    fill=False, edgecolor="black",
#                                    alpha=1,
#                                    transform=ccrs.PlateCarree()))

    # Add marker or text to denote mine location
    ax.plot(lon_mines[mine], lat_mines[mine], 'wo', markersize=7, transform = ccrs.PlateCarree())
    ax.text(lon_mines[mine]+0.06, lat_mines[mine]-0.06, mines_short[mine],
     transform=ccrs.PlateCarree(), color="white", fontsize=12)

ax0.set_aspect('auto')
ax.set_aspect('auto')
         
#%%
ch4_mine={}
ch4_bg_mine={}
for mine in mines:

    ch4_mine[mine] = ch4.sel(lon=slice(site_lons[mine][0], site_lons[mine][1]), 
                            lat=slice(site_lats[mine][0], site_lats[mine][1]))

    ch4_bg_mine[mine] = ch4.sel(lon=slice(bg_site_lons[mine][0], bg_site_lons[mine][1]), 
                            lat=slice(bg_site_lats[mine][0], bg_site_lats[mine][1]))


# Might need to define different background for each mine. Try that. 


#%%
fname_met = "/home/mlunt/datastore/MOYA/oz/GEOSFP.daily_met_" + year + "_2x25.OZ.nc"
ds_met = open_ds(fname_met)

fname_met2 = "/home/mlunt/datastore/MOYA/oz/GEOSFP.daily_met_2017_025x03125.OZ.nc"
ds_met2 = open_ds(fname_met2)

#%%
#mines = ["Moranbah", "Coppabella", "Hail_creek", "Capcoal"]

W_mine={}
for mine in mines:
    W_mine[mine]  = np.sqrt((site_lons[mine][1] - site_lons[mine][0]) * 100
                            * (site_lats[mine][1] - site_lats[mine][0]) *100)
#W_mine={"Moranbah": 14,
#        "Coppabella": 5,
#        "Hail_creek": 8,
#        "Capcoal": 12}

delta_ch4_mine={}
delta_ch4_mine_mn={}
for mine in mines:
    delta_ch4_mine[mine] = ch4_mine[mine] - ch4_bg_mine[mine].mean()
    delta_ch4_mine_mn[mine] = delta_ch4_mine[mine].mean() 

#delta_ch4_mine = {"Moranbah": ch4_mor.mean()-ch4_bg.mean(),
#                  "Coppabella": ch4_sw.mean()-ch4_bg.mean(),
#                  "Hail_creek": ch4_hc.mean()-ch4_bg.mean(),
#                  "Capcoal": ch4_gt.mean()-ch4_bg.mean(),}

lon_site = 148.0
lat_site = -21.9


U_site={}
V_site={}
ps_site={}
for mine in mines:

    lon_site = lon_mines[mine]
    lat_site = lat_mines[mine]
    U_site[mine] = ds_met2.U.sel(lon=lon_site, lat=lat_site, method="nearest")
    V_site[mine] = ds_met2.V.sel(lon=lon_site, lat=lat_site, method="nearest")
    ps_site[mine] = ds_met.P_surf.sel(lon=lon_site, lat=lat_site, method="nearest")
# Calculate emissions
#delta_ch4 = ch4.max() - np.percentile(ch4, 1)
#delta_ch4 = ch4_gt.mean() - ch4_bg.mean()
# Need to properly select areas of interest for plume and background. 

W = 10 # 10x10 km square region
M_atm = 0.029  # Mass atmosphere kg/mol
M_ch4 = 0.016 # Mass CH4 kg/mol
g = 9.8  # m/s2
p = 1000 # hPa

U = 5 # km hour-1 Uniform wind speed
wind_speed_mean={}
wind_speed_std={}
ps_mean={}
wdir_site={}
wspeed_site={}
for mine in mines:
    wind_speed_mean[mine] = np.mean(np.sqrt(U_site[mine]**2 + V_site[mine]**2))
    wind_speed_std[mine] = np.std(np.sqrt(U_site[mine]**2 + V_site[mine]**2))
    ps_mean[mine] = np.mean(ps_site[mine])
    wspeed_site[mine]  = np.sqrt(U_site[mine]**2 + V_site[mine]**2)
    wdir_site[mine] = np.arctan2(V_site[mine],U_site[mine])*180/np.pi
#wind_dir_mn = np.median(wind_dir)*180./np.pi


#wind_speed_mean = 1.1  # m s-1
#ps_mean = np.mean(ps_site)
# Q in tonnes per hour output

Q_mine={}
uncert_mine={}
for mine in mines:
#Q = delta_ch4*1.e-9 * M_ch4/M_atm * U*1000/60./60. * W*1.e3 * p*100/9.8
    Q = (delta_ch4_mine_mn[mine]*1.e-9 * M_ch4/M_atm * wind_speed_mean[mine]/2. *
         W_mine[mine]*1.e3 * ps_mean[mine]*100/g)
    # Q in kg/s
    uncert = np.sqrt( (delta_ch4_mine[mine].std()/delta_ch4_mine_mn[mine])**2 +
                        (wspeed_site[mine].std()/wind_speed_mean[mine])**2 )

    Q_out = Q/1.e3 * 60**2   # tonnes hr-1
    Q_year = Q_out*24.*365. # tonnes year-1

    Q_mine[mine] = Q_year*gwp/1.e6 # MtCO2-eq yr-1 

    uncert_mine[mine] = uncert
# QLD Fugitive emission from fuels = 698 ktonnes, Energy total = 714 ktonnes
# So 190/698 = 27% of QLD's total energy emissions from one mine? 
# Big uncertainty here is W - is this the size of the plume or the size of the emission area?
# Potentially about 10% of all coal mining emissions in Oz? 

for mine in mines:
    abs_uncert = Q_mine[mine] * uncert_mine[mine]
    print(mine, abs_uncert)

#%%
# Try cross-sectional flux method for hail creek
delta_omega={}
Q_xs_mine={}
IME={}
Q_ime={}
for mine in mines:
    delta_omega[mine] = M_ch4/M_atm * ps_mean[mine]*100./g * delta_ch4_mine[mine]*1.e-9

    Q_xs = delta_omega[mine][:,0].sum()*wind_speed_mean[mine] # kg s-1
    Q_xs_mine[mine]  = Q_xs/1.e3*60**2*24*365*25/1.e6 

    IME[mine] = (delta_omega[mine]*(2*1.e3)**2).sum()
    Q_ime[mine] = IME[mine]*wind_speed_mean[mine]/(W_mine[mine]*1.e3)/1.e3*60*60*24*365*25/1.e6 

# Buchwitz method
# Q = delta_x * CF
# CF = M * M_exp * L * V * C
M_buch = 5.345e-9 # MtCH4 km^-2 ppb^-1 
M_exp = ps_mean[mine]/1013
L_buch = W*1. # km
#V_buch = 1.1 # m s-1
V_buch = 4.2 
C = 2.

#Q_buch  = delta_ch4 * M_buch*M_exp*L_buch*V_buch/1.e3*60*60*24.*365.*C

# C factor is basically the difference between the two results whne using same length and wind speed 
# and surface pressure. 1.72/2.


#%%
"""
Q_pseudo = 1./25.*1e9 /365./24/60/60.  # MtCO2-eq yr-1
delta_x = M_atm/M_ch4 * g / (wind_speed_mean/3) / (ps_mean*100) / (W*2.*1e3) * Q_pseudo

#%%
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