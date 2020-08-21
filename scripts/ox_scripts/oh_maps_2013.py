# /usr/bin/env python2.7
# coding: utf-8

import matplotlib
matplotlib.use('agg')  # Use Agg backend for use with jasmin LOTUS

import cf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid

# Declare paths for file location
# 2014
#paths = {'Con': '../../nc_files/u-bt034/', '3A1': '../../nc_files/u-bt090/', '3A2': '../../nc_files/u-bt091/', '3A3': '../../nc_files/u-bt092/', '3A4': '../../nc_files/u-bt093/'}
# 2013
paths = {'Con': '../../nc_files/u-bt341/', '3A1': '../../nc_files/u-bt342/', '3A2': '../../nc_files/u-bt343/', '3A3': '../../nc_files/u-bt344/', '3A4': '../../nc_files/u-bt345/'}

pp_paths = {'Con': '../../mass_extracts/pp_files/u-bt034/u-bt034_stash_', '3A1': '../../mass_extracts/pp_files/u-bt090/u-bt090_stash_', '3A2': '../../mass_extracts/pp_files/u-bt091/u-bt091_stash_', '3A3': '../../mass_extracts/pp_files/u-bt092/u-bt092_stash_', '3A4': '../../mass_extracts/pp_files/u-bt093/u-bt093_stash_'}

def load_data(stash, experiments=['Con','3A1','3A2','3A3','3A4']):
    datalist = {}
    for experiment in experiments:
        try:
            path = paths[experiment] + stash + '/' + stash + '_v3.nc'
            dataset = cf.read(path)[0]
        except:
            print('No v2 file found')
            try:
                path = paths[experiment] + stash + '/' + stash + '_v2.nc'
                dataset = cf.read(path)[0]
            except:
                path = paths[experiment] + stash + '/' + stash + '.nc'
                dataset = cf.read(path)[0]
        datalist[experiment] = dataset
    return datalist

def load_pp_data(stash, experiments=['Con','3A1','3A2','3A3','3A4']):
    '''
    Loads a dictionary for cf.Field objects for a specific stash
    Loads data directly from pp files
    '''
    datalist={}
    for experiment in experiments:
        path = pp_paths[experiment] + stash + '/2014/*.pp'
        dataset = cf.read(path, verbose=True)[0]
        datalist[experiment] = dataset
    return datalist

def new_coords(lon, lat):
    '''
    Adjusts the coordinates by +0.5 lon and +0.5 lat
    This changes the coordinate from bottom left to center of box
    For use with pcolormesh maps
    '''
    lon_extend = np.zeros(lon.size+2)  # Extend by 2
    lon_extend[1:-1] = lon  # Fill with original lons
    lon_extend[0] = lon[0]-np.diff(lon)[0]  # Extend endpoints
    lon_extend[-1] = lon[-1]+np.diff(lon)[-1]  # Extend endpoints
    lon_p_mid = lon_extend[:-1] + 0.5*(np.diff(lon_extend))  # Calculate midpoints

    lat_extend = np.zeros(lat.size+2)  # Extend by 2
    lat_extend[1:-1] = lat  # Fill with original values
    lat_extend[0] = lat[0] - np.diff(lat)[0]  # Extend endpoints
    lat_extend[-1] = lat[-1] + np.diff(lat)[-1]  # Extend endpoints
    lat_p_mid = lat_extend[:-1] + 0.5*(np.diff(lat_extend))  # Calculate midpoints
    return lon_p_mid, lat_p_mid

def get_data(dataset, Mr):
    '''
    Extract separated numpy arrays of surface, lon, and lat from cf.Field object
    Adds central cyclic point to longitude to remove stripe
    '''
    surface = np.nanmean(dataset[75:135,0,:,:]*(28.97/Mr)*1e12, axis=0).squeeze()
    lon = dataset.coord('longitude').array
    lat = dataset.coord('latitude').array
    surface, lon = add_cyclic_point(surface, lon)
    lon, lat = new_coords(lon, lat)
    return surface, lon, lat

if __name__=="__main__":
    # Prepare initial parameters
    colorlist = {'Con':'k', '3A1':'b', '3A2':'r', '3A3':'g', '3A4':'orange'}
    experiment_order = ['Con','3A1','3A2','3A3','3A4']

    # Load in required data
    oh_datalist = load_data('34081')

    # Axes parameters
    axes_class = (GeoAxes, dict(map_projection=ccrs.Robinson()))  # Setup parameters. Axes class is cartopy GeoAxes, with Robinson projection

    # Set up figure
    fig = plt.figure(figsize=(14, 7), dpi=240)
    ax1 = plt.subplot2grid((2, 6), loc=(0, 2), colspan=2, projection=ccrs.Robinson())
    ax2 = plt.subplot2grid((2, 6), loc=(0, 0), colspan=2, projection=ccrs.Robinson())
    ax3 = plt.subplot2grid((2, 6), loc=(1, 1), colspan=2, projection=ccrs.Robinson())
    ax4 = plt.subplot2grid((2, 6), loc=(1, 3), colspan=2, projection=ccrs.Robinson())
    ax5 = plt.subplot2grid((2, 6), loc=(0, 4), colspan=2, projection=ccrs.Robinson())
    axes = [ax1, ax2, ax3, ax4, ax5]
    
    for i, experiment in enumerate(['Con', '3A1','3A2','3A3','3A4']):
        axes[i].coastlines()  # Draw coastlines

        # Get data
        surface, lon, lat = get_data(oh_datalist[experiment], 17.01)
    
        # Draw map data. Give handle p for colorbar use
        p = axes[i].pcolormesh(lon, lat, surface, cmap='Reds', vmin=0, vmax=0.4, transform=ccrs.PlateCarree())
        axes[i].set_title(experiment)
    
    cax, kw = matplotlib.colorbar.make_axes(axes, location='bottom', pad=0.1, shrink=0.7)    
    cb = plt.colorbar(p, cax=cax, extend='both', **kw)
    cb.set_label('OH conc / pptv')
    plt.suptitle('Surface OH Concentration')
    plt.savefig('surf_oh_map_2013.png')

    # Set up figure
    fig = plt.figure(figsize=(12,7), dpi=240)
    ax1 = plt.subplot(221, projection=ccrs.Robinson())
    ax2 = plt.subplot(222, projection=ccrs.Robinson())
    ax3 = plt.subplot(223, projection=ccrs.Robinson())
    ax4 = plt.subplot(224, projection=ccrs.Robinson())
    axes = [ax1, ax2, ax3, ax4]

    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        axes[i].coastlines()  # Draw coastlines

        # Get data
        surface, lon, lat = get_data(oh_datalist[experiment], 17.01)
        con_surface, _, _ = get_data(oh_datalist['Con'], 17.01)
        diff = surface - con_surface
    
        # Draw map data. Give handle p for colorbar use
        p = axes[i].pcolormesh(lon, lat, diff, cmap='RdBu_r', vmin=-0.05, vmax=0.05, transform=ccrs.PlateCarree())
        axes[i].set_title(experiment)
    
    cax, kw = matplotlib.colorbar.make_axes(axes, location='bottom', pad=0.1, shrink=0.7)    
    cb = plt.colorbar(p, cax=cax, extend='both', **kw)
    cb.set_label('OH diff / pptv')
    plt.suptitle('Mean OH surface diff')
    plt.savefig('surf_oh_diff_map_2013.png')



