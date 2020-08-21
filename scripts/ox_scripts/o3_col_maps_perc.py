#! /usr/bin/env python2.7
import matplotlib
matplotlib.use('agg')

import cf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid

paths = {'Con': '../../nc_files/u-bt034/', '3A1': '../../nc_files/u-bt090/', '3A2': '../../nc_files/u-bt091/', '3A3': '../../nc_files/u-bt092/', '3A4': '../../nc_files/u-bt093/'}
pp_paths = {'Con': '../../mass_extracts/pp_files/u-bt034/u-bt034_stash_', '3A1': '../../mass_extracts/pp_files/u-bt090/u-bt090_stash_', '3A2': '../../mass_extracts/pp_files/u-bt091/u-bt091_stash_', '3A3': '../../mass_extracts/pp_files/u-bt092/u-bt092_stash_', '3A4': '../../mass_extracts/pp_files/u-bt093/u-bt093_stash_'}

def load_data(stash, experiments=['Con','3A1','3A2','3A3','3A4']):
    datalist = {}
    for experiment in experiments:
        try:
            path = paths[experiment] + stash + '/' + stash + '_v4.nc'
            dataset = cf.read(path)[0]
        except:
            print('No v2 file found')
            '''
            try:
                path = paths[experiment] + stash + '/' + stash + '_v2.nc'
                dataset = cf.read(path)[0]
            except:
                path = paths[experiment] + stash + '/' + stash + '.nc'
                dataset = cf.read(path)[0]               
            '''
        datalist[experiment] = dataset
    return datalist

def load_pp_data(stash, experiments=['Con','3A1','3A2','3A3','3A4']):
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

def get_data(dataset, exp, Mr):
    '''
    Extract separated numpy arrays of surface, lon, and lat from cf.Field object
    Adds central cyclic point to longitude to remove stripe
    '''
    surface = dataset[exp][75:135,:,:,:].array * airmass[exp][75:135,:,:,:].array
    surface *= trop_mask['Con'][75:135,:,:,:].array
    surface = np.nansum(surface, axis=1)
    surface = surface*1000/area
    surface = surface*1000/Mr
    surface = np.nanmean(surface, axis=0)
    lon = dataset[exp].coord('longitude').array
    lat = dataset[exp].coord('latitude').array
    surface, lon = add_cyclic_point(surface, lon)
    lon, lat = new_coords(lon, lat)
    return surface*2.241, lon, lat

if __name__=="__main__":

    # Prepare initial parameters
    colorlist = {'Con':'k', '3A1':'b', '3A2':'r', '3A3':'g', '3A4':'orange'}
    experiment_order = ['Con','3A1','3A2','3A3','3A4']

    # Load in required data
    o3_datalist = load_data('34001')
    #no2_datalist= load_data('34996')
    airmass = load_data('50063')
    trop_mask = load_data('50064', experiments=['Con'])    
    area = cf.read('../../area/*.nc')[0][:].squeeze()
    
    # Axes parameters
    axes_class = (GeoAxes, dict(map_projection=ccrs.Robinson()))  # Setup parameters. Axes class is cartopy GeoAxes, with Robinson projection

    # Set up figure
    fig = plt.figure(figsize=(12,7), dpi=240)
    axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(2,2), axes_pad=0.6, cbar_location='right', cbar_mode='single', cbar_pad=0.2, cbar_size='2%', label_mode='')
    # nrows_ncols - Regular grid of subplots, axes_pad - pad between axes, label_mode - important.

    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        axgr[i].coastlines()  # Draw coastlines

        # Get data
        surface, lon, lat = get_data(o3_datalist, experiment, 48.00) 
        con_surface, _, _ = get_data(o3_datalist, 'Con', 48.00) 
        diff = surface - con_surface
        perc = diff / surface * 100
        rf = diff * 0.042  # Dobson Unit
        print(area.shape, diff.shape)
        rf = np.average(rf[:,:-1], weights=area)
        total_perc = np.average(perc[:,:-1], weights=area)
        # Draw map data. Give handle p for colorbar use
        #p = axgr[i].pcolormesh(lon, lat, diff, cmap='RdBu_r', vmin=-2, vmax=2, transform=ccrs.PlateCarree())
        p = axgr[i].pcolormesh(lon, lat, perc, cmap='RdBu_r', vmin=-10, vmax=10, transform=ccrs.PlateCarree())
        #axgr[i].set_title(experiment + ': %.3f Wm-2' % rf)
        axgr[i].set_title(experiment + ': %.2f%%' % total_perc)

    cb = plt.colorbar(p, cax=axgr.cbar_axes[0])
    cb.set_label('Column Diff / %')
    plt.suptitle('Difference in Tropospheric O3 column')
    plt.savefig('surf_o3_col_perc_map_new.png')
    plt.show()

