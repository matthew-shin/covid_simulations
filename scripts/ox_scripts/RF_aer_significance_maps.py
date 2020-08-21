#! /usr/bin/env python2.7
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams["hatch.linewidth"] = 0.5

import cf
import numpy as np
import pandas as pd
import netCDF4 as nc
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

def load_F_data(year, version="F", experiments=['Con','3A1','3A2','3A3','3A4']):
    datalist = {}
    for experiment in experiments:
        if year==2014:
            path = paths_2014[experiment]
        if year==2013:
            path = paths_2013[experiment]
        if year==2012:
            path = paths_2012[experiment]
        if version=="F":
            dataset, _ = F(path, 76, 136)
        elif version=="F_clean":
            dataset, _ = F_clean(path, 76, 136)
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

def F(path,start,finish):
    '''
    Total net incoming radiation : In - Out
    Returns the total map and area-weighted mean time series
    '''
    sw_in=nc.Dataset(path+'/01207/01207_v4.nc')
    sw_in=sw_in.variables['UM_m01s01i207_vn1105'][start:finish,:,:]
    
    sw_out=nc.Dataset(path+'/01208/01208_v4.nc')
    sw_out=sw_out.variables['UM_m01s01i208_vn1105'][start:finish,:,:]
    
    lw_out=nc.Dataset(path+'/02205/02205_v4.nc')
    lw_out=lw_out.variables['UM_m01s02i205_vn1105'][start:finish,:,:]
    
    flux = sw_in - sw_out - lw_out
    
    flux_awm=np.ma.average(flux, weights=np.broadcast_to(area, flux.shape),axis=(1,2))
    
    return flux, flux_awm

def F_clean(path,start,finish):
    '''
    Total net incoming clean radiation : In - Out
    Returns the total map and area-weighted mean time series
    '''
    sw_in=nc.Dataset(path+'/01207/01207_v4.nc')
    sw_in=sw_in.variables['UM_m01s01i207_vn1105'][start:finish,:,:]
   
    sw_out_clean=nc.Dataset(path+'/01517/01517_v4.nc')
    sw_out_clean=sw_out_clean.variables['UM_m01s01i517_vn1105'][start:finish,85,:,:]
   
    lw_out_clean=nc.Dataset(path+'/02517/02517_v4.nc')
    lw_out_clean=lw_out_clean.variables['UM_m01s02i517_vn1105'][start:finish,85,:,:]
   
    flux = sw_in - sw_out_clean - lw_out_clean
   
    flux_awm=np.ma.average(flux, weights=np.broadcast_to(area, flux.shape),axis=(1,2))
   
    return flux, flux_awm

def calculate_aerosol(F, FClean):
    aerosol = {}
    for experiment in F:
        aerosol[experiment] = F[experiment] - FClean[experiment]
    return aerosol

def calculate_stats(inputArray, controlArray):
    meanArray = np.nanmean(inputArray, axis=0)
    diffArray = inputArray - controlArray
    meanDiffArray = np.nanmean(diffArray, axis=0)
    varArray = np.var(diffArray, axis=0)
    return meanArray, diffArray, meanDiffArray, varArray

def adjust_points(data, lon, lat):
    dataCyclic, lonCyclic = add_cyclic_point(data, lon)
    lonShifted, latShifted = new_coords(lonCyclic, lat)
    return dataCyclic, lonShifted, latShifted

if __name__=="__main__":

    # Prepare initial parameters
    colorlist = {'Con':'k', '3A1':'b', '3A2':'r', '3A3':'g', '3A4':'orange'}
    experiment_order = ['Con','3A1','3A2','3A3','3A4']

    area = cf.read('../../area/*.nc')[0]

    # Load in required data
    FDataset2014 = load_F_data(2014, version="F")
    FDataset2013 = load_F_data(2013, version="F")
    FDataset2012 = load_F_data(2012, version="F")

    FCleanDataset2014 = load_F_data(2014, version="F_clean")
    FCleanDataset2013 = load_F_data(2013, version="F_clean")
    FCleanDataset2012 = load_F_data(2012, version="F_clean")

    AerDataset2014 = {experiment:FDataset2014[experiment]-FCleanDataset2014[experiment] for experiment in FDataset2014}
    AerDataset2013 = {experiment:FDataset2013[experiment]-FCleanDataset2013[experiment] for experiment in FDataset2013}
    AerDataset2012 = {experiment:FDataset2012[experiment]-FCleanDataset2012[experiment] for experiment in FDataset2012}

    lon = area.coord('longitude').array
    lat = area.coord('latitude').array
    
    # Calculate the total standard deviation of three distributions by combining their deviance:
    # Total deviance = Sum(Deviance of each distribution) + Sum(Deviance of the means)
    # Deviance is n(x - xbar), or also known as variance * number of points

    # Calculate total standard deviation and total mean for all three years
    # Then calculate the upper and lower bounds given 2*standard dev (95% confidence interval)
    # If zero is contained in this bound, then we can say with 95% confidence that there is no difference!

    # Axes parameters
    axes_class = (GeoAxes, dict(map_projection=ccrs.PlateCarree()))  # Setup parameters. Axes class is cartopy GeoAxes, with PlateCarree projection

    # Set up figure
    #fig = plt.figure(figsize=(12,7), dpi=240)
    #axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(2,2), axes_pad=0.6, label_mode="")#, cbar_location='bottom', cbar_mode='single', cbar_pad=0.2, cbar_size='2%', label_mode='')
    fig = plt.figure(figsize=(12,7), dpi=300)
    ax1 = plt.subplot(221, projection=ccrs.PlateCarree())
    ax2 = plt.subplot(222, projection=ccrs.PlateCarree())
    ax3 = plt.subplot(223, projection=ccrs.PlateCarree())
    ax4 = plt.subplot(224, projection=ccrs.PlateCarree())
    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    axgr = [ax1, ax2, ax3, ax4]

    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        axgr[i].coastlines(linewidth=0.5)  # Draw coastlines

        mean2014, diff2014, meanDiff2014, var2014 = calculate_stats(AerDataset2014[experiment], AerDataset2014['Con'])
        mean2013, diff2013, meanDiff2013, var2013 = calculate_stats(AerDataset2013[experiment], AerDataset2013['Con'])
        mean2012, diff2012, meanDiff2012, var2012 = calculate_stats(AerDataset2012[experiment], AerDataset2012['Con'])
    
        totalMean = (mean2014 + mean2013 + mean2012)/3.
        totalMeanDiff = (meanDiff2014 + meanDiff2013 + meanDiff2012)/3.
    
        totalDeviance = var2014.shape[0]*var2014 \
                        + var2013.shape[0]*var2013 \
                        + var2012.shape[0]*var2012 \
                        + meanDiff2014.shape[0]*(meanDiff2014 - totalMeanDiff)**2. \
                        + meanDiff2013.shape[0]*(meanDiff2013 - totalMeanDiff)**2. \
                        + meanDiff2012.shape[0]*(meanDiff2012 - totalMeanDiff)**2.
    
        totalStd = np.sqrt(totalDeviance/(var2014.shape[0]+var2013.shape[0]+var2012.shape[0]))
     
        lowerBound = totalMeanDiff - 1.96*totalStd/np.sqrt(var2014.shape[0]+var2013.shape[0]+var2012.shape[0])
        upperBound = totalMeanDiff + 1.96*totalStd/np.sqrt(var2014.shape[0]+var2013.shape[0]+var2012.shape[0])
        
        maskSignificance = np.ma.masked_where(~((lowerBound<0) & (upperBound>0)), totalMeanDiff, copy=True)

        plotTMD, plotLon, plotLat = adjust_points(totalMeanDiff, lon, lat)
        plotMask, _, _ = adjust_points(maskSignificance, lon, lat)

        p = axgr[i].pcolormesh(plotLon, plotLat, plotTMD, cmap='bwr', vmin=-3, vmax=+3, transform=ccrs.PlateCarree())
        axgr[i].pcolor(plotLon, plotLat, plotMask, hatch='//////', alpha=0.)
        axgr[i].set_title(experiment[1:])
        #im = ax.pcolormesh(totalMeanDiff)
        #ax.pcolor(testMask, hatch='xx', alpha=0)
        #plt.show()
    cax, kw = matplotlib.colorbar.make_axes(axgr, location='bottom', pad=0.1, shrink=0.7, aspect=30)
    cb = plt.colorbar(p, cax=cax, extend='both', **kw)
    cb.set_label('RF / Wm-2')
    plt.suptitle('Aerosol-only TOA RF')
    plt.savefig('Aerosol_95_RF_maps.jpg')
    #plt.show()

