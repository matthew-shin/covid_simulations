# /usr/bin/env python2.7
# coding: utf-8

import matplotlib
matplotlib.use('agg')  # Use Agg backend for use with jasmin LOTUS

import cf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import netCDF4 as nc
from cartopy.util import add_cyclic_point
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid

# Declare paths for file location
# emiss 2014
paths_2014 = {'Con': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/Control-14/BC_*.nc', '3A1': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/E3A1-14/E3A1_BC_*.nc', '3A2': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/E3A2-14/E3A2_BC_*.nc', '3A3': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/E3A3-14/E3A3_BC_*.nc', '3A4': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/E3A5-14/E3A5_BC_*.nc'}
# emiss 2013
paths_2013 = {'Con': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/Control-13/BC_*.nc', '3A1': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/E3A1-13/E3A1_BC_*.nc', '3A2': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/E3A2-13/E3A2_BC_*.nc', '3A3': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/E3A3-13/E3A3_BC_*.nc', '3A4': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/E3A5-13/E3A5_BC_*.nc'}
# emiss 2012
paths_2012 = {'Con': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/Control-12/BC_*.nc', '3A1': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/E3A1-12/E3A1_BC_*.nc', '3A2': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/E3A2-12/E3A2_BC_*.nc', '3A3': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/E3A3-12/E3A3_BC_*.nc', '3A4': '../../../../yms23/UKESM/emissions_preparation/COVID_ems/E3A5-12/E3A5_BC_*.nc'}

pp_paths = {'Con': '../../mass_extracts/pp_files/u-bt034/u-bt034_stash_', '3A1': '../../mass_extracts/pp_files/u-bt090/u-bt090_stash_', '3A2': '../../mass_extracts/pp_files/u-bt091/u-bt091_stash_', '3A3': '../../mass_extracts/pp_files/u-bt092/u-bt092_stash_', '3A4': '../../mass_extracts/pp_files/u-bt093/u-bt093_stash_'}

def load_data(stash, year, experiments=['Con','3A1','3A2','3A3','3A4']):
    datalist = {}
    for experiment in experiments:
        if year==2014:
            path = paths_2014[experiment]
        if year==2013:
            path = paths_2013[experiment]
        if year==2012:
            path = paths_2012[experiment]
        dataset = cf.read(path)
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

def get_data(dataset, Mr, year):
    '''
    Extract separated numpy arrays of surface, lon, and lat from cf.Field object
    Adds central cyclic point to longitude to remove stripe
    '''
    if year==2014:
        first, second = 26, 29
    if year==2013:
        first, second = 14, 17
    if year==2012:
        first, second = 26, 29
    surface = np.array([i[first:second,:,:,:].array for i in dataset])  # Extract sectors
    surface = np.nansum(surface, axis=0).squeeze()  # Sum sectors
    surface = np.nanmean(surface, axis=0)  # Mean time
    lon = dataset[0].coord('longitude').array
    lat = dataset[0].coord('latitude').array
    surface, lon = add_cyclic_point(surface, lon)
    lon, lat = new_coords(lon, lat)
    return surface, lon, lat

if __name__=="__main__":
    # Prepare initial parameters
    colorlist = {'Con':'k', '3A1':'b', '3A2':'r', '3A3':'g', '3A4':'orange'}
    experiment_order = ['Con','3A1','3A2','3A3','3A4']

    # Load in required data
    data_2014 = load_data('34001', 2014)
    data_2013 = load_data('34001', 2013)
    data_2012 = load_data('34001', 2012)
    
    area = cf.read('../../area/areacella*.nc')[0].array
    nat = nc.Dataset('/home/users/yms23/ukca_emiss_SO2_nat.nc').variables['emissions_SO2_nat_kgSO2'][:] 
    
    # Axes parameters
    axes_class = (GeoAxes, dict(map_projection=ccrs.PlateCarree()))  # Setup parameters. Axes class is cartopy GeoAxes, with PlateCarree projection
    '''
    # Set up figure
    fig = plt.figure(figsize=(14, 7), dpi=240)
    ax1 = plt.subplot2grid((2, 6), loc=(0, 2), colspan=2, projection=ccrs.PlateCarree())
    ax2 = plt.subplot2grid((2, 6), loc=(0, 0), colspan=2, projection=ccrs.PlateCarree())
    ax3 = plt.subplot2grid((2, 6), loc=(1, 1), colspan=2, projection=ccrs.PlateCarree())
    ax4 = plt.subplot2grid((2, 6), loc=(1, 3), colspan=2, projection=ccrs.PlateCarree())
    ax5 = plt.subplot2grid((2, 6), loc=(0, 4), colspan=2, projection=ccrs.PlateCarree())
    axes = [ax1, ax2, ax3, ax4, ax5]
    
    for i, experiment in enumerate(['Con', '3A1','3A2','3A3','3A4']):
        axes[i].coastlines()  # Draw coastlines

        # Get data
        surface, lon, lat = get_data(o3_datalist[experiment], 48.00)
    
        # Draw map data. Give handle p for colorbar use
        p = axes[i].pcolormesh(lon, lat, surface, cmap='Reds', vmin=0, transform=ccrs.PlateCarree())
        axes[i].set_title(experiment)
    
    cax, kw = matplotlib.colorbar.make_axes(axes, location='bottom', pad=0.1, shrink=0.7)    
    cb = plt.colorbar(p, cax=cax, extend='both', **kw)
    cb.set_label('BC emissions')
    plt.suptitle('2014 surface BC emissions')
    plt.savefig('surf_BC_emissions_2014.png')
    '''
    # Set up figure
    fig = plt.figure(figsize=(10,7), dpi=240)
    ax1 = plt.subplot(221, projection=ccrs.PlateCarree())
    ax2 = plt.subplot(222, projection=ccrs.PlateCarree())
    ax3 = plt.subplot(223, projection=ccrs.PlateCarree())
    ax4 = plt.subplot(224, projection=ccrs.PlateCarree())
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    axes = [ax1, ax2, ax3, ax4]

    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        axes[i].coastlines(linewidth=0.5)  # Draw coastlines

        # Get data
        surface_2014, lon, lat = get_data(data_2014[experiment], 48.00, 2014)
        surface_2013, _, _ = get_data(data_2013[experiment], 48.00, 2013)
        surface_2012, _, _ = get_data(data_2012[experiment], 48.00, 2012)
        #urface_nat = np.sum(add_cyclic_point(nat).squeeze(), axis=0)

        surface = np.mean(np.array([surface_2014, surface_2013, surface_2012]), axis=0)
        #urface += surface_nat
        surface_total = surface[:,:-1] * area
        surface_total = np.sum(surface_total)
 
        con_2014, _, _ = get_data(data_2014['Con'], 48.00, 2014)
        con_2013, _, _ = get_data(data_2013['Con'], 48.00, 2013)
        con_2012, _, _ = get_data(data_2012['Con'], 48.00, 2012)

        con_surface = np.mean(np.array([con_2014, con_2013, con_2012]), axis=0)
        #con_surface += surface_nat
        con_total = con_surface[:,:-1] * area
        con_total = np.sum(con_total)

        diff = surface - con_surface
        diff_total = surface_total - con_total
        perc = diff_total / con_total * 100
    

        # Diff means or mean diffs
        
        

        # Draw map data. Give handle p for colorbar use
        p = axes[i].pcolormesh(lon, lat, diff, cmap='RdBu_r', vmin=-2e-12, vmax=2e-12, transform=ccrs.PlateCarree())
        axes[i].set_title(experiment[1:] + ': %.2f %%' % (perc))
        #print((np.sum((surface_2014+surface_nat)[:,:-1]*area), np.sum((con_2014+surface_nat)[:,:-1]*area), np.sum((surface_2013+surface_nat)[:,:-1]*area), np.sum((con_2013+surface_nat)[:,:-1]*area), np.sum((surface_2012+surface_nat)[:,:-1]*area), np.sum((con_2012+surface_nat)[:,:-1]*area)))
    

    cax, kw = matplotlib.colorbar.make_axes(axes, location='bottom', pad=0.1, shrink=0.7)
    cb = plt.colorbar(p, cax=cax, extend='both', **kw)
    cb.set_label('BC perturbation / kg m-2 s-1')
    plt.suptitle('BC surface emissions difference')
    plt.savefig('surf_BC_emisssions_diff_new_new.png')



