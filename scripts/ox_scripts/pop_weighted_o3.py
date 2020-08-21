# /usr/bin/env python2.7
# coding: utf-8

import matplotlib
#matplotlib.use('agg')  # Use Agg backend for use with jasmin LOTUS

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
#paths = {'Con': '../../nc_files/u-bt341/', '3A1': '../../nc_files/u-bt342/', '3A2': '../../nc_files/u-bt343/', '3A3': '../../nc_files/u-bt344/', '3A4': '../../nc_files/u-bt345/'}
# 2012
paths = {'Con': '../../nc_files/u-bt375/', '3A1': '../../nc_files/u-bt376/', '3A2': '../../nc_files/u-bt377/', '3A3': '../../nc_files/u-bt378/', '3A4': '../../nc_files/u-bt379/'}

experiment_strings = {'3A1': '-50% transport, -50% aviation, -25% industrial', '3A2': '-50% transport, -25% aviation, -25% industrial', '3A3': '-75% transport, -50% aviation, -25% industry', '3A4': '-25% transport, -25% aviation'}

pp_paths = {'Con': '../../mass_extracts/pp_files/u-bt034/u-bt034_stash_', '3A1': '../../mass_extracts/pp_files/u-bt090/u-bt090_stash_', '3A2': '../../mass_extracts/pp_files/u-bt091/u-bt091_stash_', '3A3': '../../mass_extracts/pp_files/u-bt092/u-bt092_stash_', '3A4': '../../mass_extracts/pp_files/u-bt093/u-bt093_stash_'}

def load_data(stash, experiments=['Con','3A1','3A2','3A3','3A4']):
    '''
    Loads a dictionary of cf.Field objects for a specific stash
    Loads data from the NetCDF4 files
    '''
    datalist = {}
    for experiment in experiments:
        print(experiment)
        try:
            path = paths[experiment] + stash + '/' + stash + '_v4.nc'
            dataset = cf.read(path)[0]
        except:
            print('No v2 file found')
            '''
            path = paths[experiment] + stash + '/' + stash + '.nc'
            dataset = cf.read(path)[0]
            '''
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
    surface = np.nanmean(dataset[74:134,0,:,:],axis=0).squeeze()*(28.97/Mr)*1e9
    lon = dataset.coord('longitude').array
    lat = dataset.coord('latitude').array
    surface, lon = add_cyclic_point(surface, lon)
    #lon, lat = new_coords(lon, lat)
    return surface, lon, lat

if __name__=="__main__":
    # Prepare initial parameters
    colorlist = {'Con':'k', '3A1':'b', '3A2':'r', '3A3':'g', '3A4':'orange'}
    experiment_order = ['Con','3A1','3A2','3A3','3A4']

    # Load in required data
    o3_datalist = load_data('34001')
    no2_datalist= load_data('34996')
    #pop = cf.read('../../population/global*.nc')

    # Axes parameters
    axes_class = (GeoAxes, dict(map_projection=ccrs.OSGB()))  # Setup parameters. Axes class is cartopy GeoAxes, with Robinson projection

    #fig = plt.figure()
    #ax = plt.subplot(111, projection=ccrs.OSGB())
    #ax.coastlines(resolution='50m')
    #population = np.nanmean(pop[0][158:161,...].array, axis=0).squeeze()
    #lat = pop[0].coord('dim1').array
    #lon = pop[0].coord('dim2').array
    #population, lon = add_cyclic_point(population, lon)
    #lon, lat = new_coords(lon, lat)
    #ax.pcolormesh(lon, lat, population, cmap='Reds', transform=ccrs.PlateCarree())
    #ax.set_extent([-10,5,50,60], ccrs.PlateCarree())
    #plt.savefig('population.png')
    #plt.show()
    '''
    fig = plt.figure()
    ax = plt.subplot(111, projection=ccrs.OSGB())
    ax.coastlines(resolution='50m')
    surface, lon, lat = get_data(o3_datalist['3A1'], 48.00)
    ax.imshow(surface)
    plt.show()
    '''
    '''
    # Set up figure
    fig = plt.figure(figsize=(10,12), dpi=300)
    axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(2,2), axes_pad=0.8, cbar_location='right', cbar_mode='single', cbar_pad=0.2, cbar_size='2%', label_mode='')
    # nrows_ncols - Regular grid of subplots, axes_pad - pad between axes, label_mode - important. 

    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        axgr[i].coastlines(resolution='50m', color='gray', linewidth=1)  # Draw coastlines

        # Get data
        surface, lon, lat = get_data(o3_datalist[experiment], 48.00)
        con_surface, _, _ = get_data(o3_datalist['Con'], 48.00)
        diff = surface - con_surface
        perc = diff / con_surface * 100

        #mask = np.zeros(surface.shape)
        #mask[112:120,0:1] = 1.
        #mask[112:120,-4:] = 1.
        #mask[115:120,-5] = 1.
        #indices = np.nonzero(mask)
        """
        surface_pop = np.average(surface[indices], weights=population[indices])
        con_pop = np.average(con_surface[indices], weights=population[indices])
        """
        #ma = np.ma.MaskedArray(surface*mask, mask=[mask==0])
        #ma_con = np.ma.MaskedArray(con_surface*mask, mask=[mask==0])
        #surface_pop = np.ma.average(ma, weights=population)
        #con_pop = np.ma.average(ma_con, weights=population)
        #diff_pop = surface_pop - con_pop
        #perc_pop = diff_pop / con_pop * 100

        # Draw map data. Give handle p for colorbar use
        #p = axgr[i].pcolormesh(lon, lat, perc*mask, cmap='RdBu_r', vmin=-10, vmax=10, transform=ccrs.PlateCarree())
        p = axgr[i].contourf(lon, lat, perc, 30, cmap='RdBu_r', vmin=-10, vmax=10, transform=ccrs.PlateCarree())
        axgr[i].set_extent([-10, 5, 50, 60], ccrs.PlateCarree())
        #axgr[i].set_title(experiment + '\n Population weighted mean: %.3f ppbv' % diff_pop + '\n Percentage difference: %.3f%%' % perc_pop)
        axgr[i].set_title(experiment[1:] + ' - 2014\n' + experiment_strings[experiment])
        
    cb = plt.colorbar(p, cax=axgr.cbar_axes[0])
    cb.set_label('O3 diff / %')
    plt.suptitle('Mean surface O3 difference')
    plt.savefig('o3_UK_2014.png')
    #plt.show()
    
     
    # Set up figure
    fig = plt.figure(figsize=(10,12), dpi=300)
    axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(2,2), axes_pad=0.8, cbar_location='right', cbar_mode='single', cbar_pad=0.2, cbar_size='2%', label_mode='')
    # nrows_ncols - Regular grid of subplots, axes_pad - pad between axes, label_mode - important.

    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        axgr[i].coastlines(resolution='50m', color='gray', linewidth=1)  # Draw coastlines

        # Get data
        surface, lon, lat = get_data(no2_datalist[experiment], 46.00)
        con_surface, _, _ = get_data(no2_datalist['Con'], 46.00)
        diff = surface - con_surface
        perc = diff / con_surface * 100

        #mask = np.zeros(surface.shape)
        #mask[112:120,0:1] = 1.
        #mask[112:120,-4:] = 1.
        #mask[115:120,-5] = 1.
        #indices = np.nonzero(mask)
        """
        surface_pop = np.average(surface[indices], weights=population[indices])
        con_pop = np.average(con_surface[indices], weights=population[indices])
        """
        #ma = np.ma.MaskedArray(surface*mask, mask=[mask==0])
        #ma_con = np.ma.MaskedArray(con_surface*mask, mask=[mask==0])
        #surface_pop = np.ma.average(ma, weights=population)
        #con_pop = np.ma.average(ma_con, weights=population)
        #diff_pop = surface_pop - con_pop
        #perc_pop = diff_pop / con_pop * 100

        # Draw map data. Give handle p for colorbar use
        #p = axgr[i].pcolormesh(lon, lat, perc*mask, cmap='RdBu_r', vmin=-40, vmax=40, transform=ccrs.PlateCarree())
        p = axgr[i].contourf(lon, lat, perc, 30, cmap='RdBu_r', vmin=-10, vmax=10, transform=ccrs.PlateCarree())

        axgr[i].set_extent([-10, 5, 50, 60], ccrs.PlateCarree())
        #axgr[i].set_title(experiment + '\n Population weighted mean: %.3f ppbv' % diff_pop + '\n Percentage difference: %.3f%%' % perc_pop)
        axgr[i].set_title(experiment[1:] + ' - 2014\n' + experiment_strings[experiment])

    cb = plt.colorbar(p, cax=axgr.cbar_axes[0])
    cb.set_label('NO2 diff / %')
    plt.suptitle('Mean surface NO2 difference')
    plt.savefig('no2_UK_2014.png')
    '''
###
    axes_class = (GeoAxes, dict(map_projection=ccrs.Robinson()))  # Setup parameters. Axes class is cartopy GeoAxes, with Robinson projection
    # Set up figure
    fig = plt.figure(figsize=(13,12), dpi=300)
    axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(2,2), axes_pad=0.8, cbar_location='right', cbar_mode='single', cbar_pad=0.2, cbar_size='2%', label_mode='')
    # nrows_ncols - Regular grid of subplots, axes_pad - pad between axes, label_mode - important.

    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        axgr[i].coastlines(resolution='50m', color='gray', linewidth=1)  # Draw coastlines

        # Get data
        surface, lon, lat = get_data(o3_datalist[experiment], 48.00)
        con_surface, _, _ = get_data(o3_datalist['Con'], 48.00)
        diff = surface - con_surface
        perc = diff / con_surface * 100

        #mask = np.zeros(surface.shape)
        #mask[100:123,-8:] = 1.
        #mask[100:123,:17] = 1.
        #indices = np.nonzero(mask)
        """
        surface_pop = np.average(surface[indices], weights=population[indices])
        con_pop = np.average(con_surface[indices], weights=population[indices])
        """
        #ma = np.ma.MaskedArray(surface*mask, mask=[mask==0])
        #ma_con = np.ma.MaskedArray(con_surface*mask, mask=[mask==0])
        #surface_pop = np.ma.average(ma, weights=population)
        #con_pop = np.ma.average(ma_con, weights=population)
        #diff_pop = surface_pop - con_pop
        #perc_pop = diff_pop / con_pop * 100

        # Draw map data. Give handle p for colorbar use
        #p = axgr[i].pcolormesh(lon, lat, perc*mask, cmap='RdBu_r', vmin=-10, vmax=10, transform=ccrs.PlateCarree())
        p = axgr[i].contourf(lon, lat, perc, np.arange(-10,10,0.5), cmap='RdBu_r', vmin=-10, vmax=10, transform=ccrs.PlateCarree(), extend='both')

        axgr[i].set_extent([-15, 30, 35, 65], ccrs.PlateCarree())
        #axgr[i].set_title(experiment + '\n Population weighted mean: %.3f ppbv' % diff_pop + '\n Percentage difference: %.3f%%' % perc_pop)
        axgr[i].set_title(experiment[1:] + ' - 2012\n' + experiment_strings[experiment])

    cb = plt.colorbar(p, cax=axgr.cbar_axes[0])
    cb.set_label('O3 diff / %')
    axgr[0].text(0.5, 0.9, 'Change in O3 levels due to COVID-19 lockdown', transform=fig.transFigure, horizontalalignment='center', fontsize=14)
    plt.savefig('o3_EU_2012.png')
    #plt.show()
###
    axes_class = (GeoAxes, dict(map_projection=ccrs.Robinson()))  # Setup parameters. Axes class is cartopy GeoAxes, with Robinson projection
    # Set up figure
    fig = plt.figure(figsize=(13,12), dpi=300)
    axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(2,2), axes_pad=0.8, cbar_location='right', cbar_mode='single', cbar_pad=0.2, cbar_size='2%', label_mode='')
    # nrows_ncols - Regular grid of subplots, axes_pad - pad between axes, label_mode - important.

    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        axgr[i].coastlines(resolution='50m', color='gray', linewidth=1)  # Draw coastlines

        # Get data
        surface, lon, lat = get_data(no2_datalist[experiment], 46.00)
        con_surface, _, _ = get_data(no2_datalist['Con'], 46.00)
        diff = surface - con_surface
        perc = diff / con_surface * 100

        #mask = np.zeros(surface.shape)
        #mask[100:123,-8:] = 1.
        #mask[100:123,:17] = 1.
        #indices = np.nonzero(mask)
        """
        surface_pop = np.average(surface[indices], weights=population[indices])
        con_pop = np.average(con_surface[indices], weights=population[indices])
        """
        #ma = np.ma.MaskedArray(surface*mask, mask=[mask==0])
        #ma_con = np.ma.MaskedArray(con_surface*mask, mask=[mask==0])
        #surface_pop = np.ma.average(ma, weights=population)
        #con_pop = np.ma.average(ma_con, weights=population)
        #diff_pop = surface_pop - con_pop
        #perc_pop = diff_pop / con_pop * 100

        # Draw map data. Give handle p for colorbar use
        #p = axgr[i].pcolormesh(lon, lat, perc*mask, cmap='RdBu_r', vmin=-50, vmax=50, transform=ccrs.PlateCarree())
        p = axgr[i].contourf(lon, lat, perc, np.arange(-50,50,2), cmap='RdBu_r', vmin=-50, vmax=50, transform=ccrs.PlateCarree(), extend='both')

        axgr[i].set_extent([-15, 30, 35, 65], ccrs.PlateCarree())
        #axgr[i].set_title(experiment + '\n Population weighted mean: %.3f ppbv' % diff_pop + '\n Percentage difference: %.3f%%' % perc_pop)
        axgr[i].set_title(experiment[1:] + ' - 2012\n' + experiment_strings[experiment])

    cb = plt.colorbar(p, cax=axgr.cbar_axes[0])
    cb.set_label('NO2 diff / %')
    axgr[0].text(0.5, 0.9, 'Change in NO2 levels due to COVID-19 lockdown', transform=fig.transFigure, horizontalalignment='center', fontsize=14)
    plt.savefig('no2_EU_2012.png')
    #plt.show()


