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

paths_2014 = {'Con': '../../nc_files/u-bt034/', '3A1': '../../nc_files/u-bt090/', '3A2': '../../nc_files/u-bt091/', '3A3': '../../nc_files/u-bt092/', '3A4': '../../nc_files/u-bt637/'}
paths_2013 = {'Con': '../../nc_files/u-bt341/', '3A1': '../../nc_files/u-bt342/', '3A2': '../../nc_files/u-bt343/', '3A3': '../../nc_files/u-bt344/', '3A4': '../../nc_files/u-bt926/'}
paths_2012 = {'Con': '../../nc_files/u-bt375/', '3A1': '../../nc_files/u-bt376/', '3A2': '../../nc_files/u-bt377/', '3A3': '../../nc_files/u-bt378/', '3A4': '../../nc_files/u-bt927/'}
pp_paths = {'Con': '../../mass_extracts/pp_files/u-bt034/u-bt034_stash_', '3A1': '../../mass_extracts/pp_files/u-bt090/u-bt090_stash_', '3A2': '../../mass_extracts/pp_files/u-bt091/u-bt091_stash_', '3A3': '../../mass_extracts/pp_files/u-bt092/u-bt092_stash_', '3A4': '../../mass_extracts/pp_files/u-bt093/u-bt093_stash_'}

def load_data(stash, year, experiments=['Con','3A1','3A2','3A3','3A4']):
    datalist = {}
    for experiment in experiments:
        if year==2014:
            path = paths_2014[experiment] + stash + '/' + stash + '_v4.nc'
        if year==2013:
            path = paths_2013[experiment] + stash + '/' + stash + '_v4.nc'
        if year==2012:
            path = paths_2012[experiment] + stash + '/' + stash + '_v4.nc'
        dataset = cf.read(path)[0]
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

def get_data(dataset, airmass, trop_mask, exp, Mr):
    '''
    Extract separated numpy arrays of surface, lon, and lat from cf.Field object
    Adds central cyclic point to longitude to remove stripe
    '''
    surface = dataset[exp][76:136,:,:,:].array * airmass[exp][76:136,:,:,:].array
    #surface *= trop_mask['Con'][76:136,:,:,:].array
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
    datalist_2014 = load_data('34001', year=2014)
    datalist_2013 = load_data('34001', year=2013)
    datalist_2012 = load_data('34001', year=2012)
    
    #no2_datalist= load_data('34996')
    airmass_2014 = load_data('50063', year=2014)
    airmass_2013 = load_data('50063', year=2013)
    airmass_2012 = load_data('50063', year=2012)

    trop_mask_2014 = 0#load_data('50064', year=2014, experiments=['Con'])
    trop_mask_2013 = 0#load_data('50064', year=2013, experiments=['Con'])
    trop_mask_2012 = 0#load_data('50064', year=2012, experiments=['Con'])

    area = cf.read('../../area/*.nc')[0][:].squeeze()
    
    # Axes parameters
    axes_class = (GeoAxes, dict(map_projection=ccrs.PlateCarree()))  # Setup parameters. Axes class is cartopy GeoAxes, with PlateCarree projection

    # Set up figure
    #ig = plt.figure(figsize=(12,7), dpi=240)
    #xgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(2,2), axes_pad=0.6, cbar_location='right', cbar_mode='single', cbar_pad=0.2, cbar_size='2%', label_mode='')
    # nrows_ncols - Regular grid of subplots, axes_pad - pad between axes, label_mode - important.

    fig = plt.figure(figsize=(12,7), dpi=300)
    ax1 = plt.subplot(221, projection=ccrs.PlateCarree())
    ax2 = plt.subplot(222, projection=ccrs.PlateCarree())
    ax3 = plt.subplot(223, projection=ccrs.PlateCarree())
    ax4 = plt.subplot(224, projection=ccrs.PlateCarree())
    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    axgr = [ax1, ax2, ax3, ax4]

    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        axgr[i].coastlines(linewidth=0.5)  # Draw coastlines

        # Get data
        surface_2014, lon, lat = get_data(datalist_2014, airmass_2014, trop_mask_2014, experiment, 48.00)
        surface_2013, _, _ = get_data(datalist_2013, airmass_2013, trop_mask_2013, experiment, 48.00)
        surface_2012, _, _ = get_data(datalist_2012, airmass_2012, trop_mask_2012, experiment, 48.00)

        con_2014, _, _ = get_data(datalist_2014, airmass_2014, trop_mask_2014, 'Con', 48.00)
        con_2013, _, _ = get_data(datalist_2013, airmass_2013, trop_mask_2013, 'Con', 48.00)
        con_2012, _, _ = get_data(datalist_2012, airmass_2012, trop_mask_2012, 'Con', 48.00)

        diff_2014 = surface_2014 - con_2014
        diff_2013 = surface_2013 - con_2013
        diff_2012 = surface_2012 - con_2012

        diff = np.mean(np.array([diff_2014, diff_2013, diff_2012]), axis=0) 
        surface = np.mean(np.array([con_2014, con_2013, con_2012]), axis=0)
        perc = diff / surface * 100

        rf = diff * 0.042
        rf_2014 = diff_2014 * 0.042  # Dobson Unit
        rf_2013 = diff_2013 * 0.042
        rf_2012 = diff_2012 * 0.042

        #print(area.shape, diff.shape)
        rf = np.average(rf[:,:-1], weights=area)
        rf_2014 = np.average(rf_2014[:,:-1], weights=area)
        rf_2013 = np.average(rf_2013[:,:-1], weights=area)
        rf_2012 = np.average(rf_2012[:,:-1], weights=area)

        #total_perc = np.average(perc[:,:-1], weights=area)
        # Draw map data. Give handle p for colorbar use
        p = axgr[i].pcolormesh(lon, lat, diff, cmap='RdBu_r', vmin=-5, vmax=5, transform=ccrs.PlateCarree())
        #p = axgr[i].pcolormesh(lon, lat, perc, cmap='RdBu_r', vmin=-10, vmax=10, transform=ccrs.PlateCarree())
        #axgr[i].set_title(experiment)
        axgr[i].set_title(experiment[1:] + ': %.3f Wm-2' % rf)
        #xgr[i].set_title(experiment + ': %.2f%%' % total_perc)
        print(experiment)
        print(rf_2014, rf_2013, rf_2012)

    cax, kw = matplotlib.colorbar.make_axes(axgr, location='bottom', pad=0.1, shrink=0.7, aspect=30)
    cb = plt.colorbar(p, cax=cax, extend='both', **kw)

    #cb = plt.colorbar(p, cax=axgr.cbar_axes[0], extend='both')
    #cb.set_label('Column Difference / %')
    cb.set_label('Column Difference / DU')
    plt.suptitle('Absolute difference in Total O3 column')
    plt.savefig('mean_surf_o3_col_total.jpg')
    plt.show()

