# -*- coding: utf-8 -*-
"""
Created on Sat May  7 08:55:29 2016

@author: andyjones
"""

import scipy as sp
import scipy.ndimage
import scipy.interpolate
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import tfl

LONDON_LONS = (-.35, +.15)
LONDON_LATS = (51.4, 51.65)

PRICE_PAID_COLUMNS = {
        0: 'id',  
        1: 'price',
        2: 'date',
        3: 'postcode',
        4: 'type',
        5: 'age',
        6: 'duration',
        7: 'paon',
        8: 'saon',
        9: 'street',
        10: 'locality', 
        11: 'town',
        12: 'local_authority',
        13: 'county',
        14: 'UNKNOWN',
        15: 'record_status'
    }

def load_price_paid():
    data = pd.read_csv('data/pp-2015.csv', header=None)
    data = data.rename(columns=PRICE_PAID_COLUMNS)
    return data
    
def load_postcodes():
    data = pd.read_csv('data/ukpostcodes.zip')
    return data.set_index('postcode')[['latitude', 'longitude']]
    
def filter_to_london(coords):
    lat_mask = (coords.latitude > LONDON_LATS[0]) & (coords.latitude < LONDON_LATS[1])    
    lon_mask = (coords.longitude > LONDON_LONS[0]) & (coords.longitude < LONDON_LONS[1])
    
    return coords[lat_mask & lon_mask]    
    
def get_price_paid_coords():
    postcodes = load_postcodes()
    price_paid = load_price_paid()[['price', 'date', 'postcode']]
    merged = pd.merge(price_paid, postcodes, left_on='postcode', right_index=True)
    
    london = filter_to_london(merged)
    return london.drop('postcode', 1)
    
def get_basemap_params(coords):
    padding = 0.01
    options = {
        'projection': 'tmerc',
        'lat_0': 50.,
        'lon_0': 0.,
        'ellps': 'WGS84',
        'urcrnrlat': coords.latitude.max() + padding,
        'llcrnrlat': coords.latitude.min() - padding,
        'urcrnrlon': coords.longitude.max() + padding,
        'llcrnrlon': coords.longitude.min() - padding,
        'resolution': 'c',
    }
    return options

def get_time_to_location(edges, locations):
    times = tfl.get_travel_times(edges, locations)
    
    index = pd.Series(times.index & locations.index)
    results = pd.concat([times.loc[index], locations.loc[index, ['longitude', 'latitude']]], 1)

    results = results.rename(columns={0: 'time'})

    return results
    
def get_resolution():
    EARTH_CIRCUMFERENCE = 40000
    RES = .1
    LAT_RES = 360*RES/EARTH_CIRCUMFERENCE
    
    CIRCUM_AT_LATITUDE = EARTH_CIRCUMFERENCE*sp.cos(sp.pi/180*LONDON_LATS[0])
    LON_RES = 360*RES/CIRCUM_AT_LATITUDE
    
    return (LON_RES, LAT_RES)
    
def get_grid_indices(coords):
    lon_res, lat_res = get_resolution()
    
    lon_index = (coords.longitude - LONDON_LONS[0])/lon_res
    lat_index = (coords.latitude - LONDON_LATS[0])/lat_res
    
    return pd.DataFrame({'x': lon_index, 'y': lat_index}).astype(int)

def get_array(coords, values):
    lon_res, lat_res = get_resolution()    
    
    indices = get_grid_indices(coords)
    indices['values'] = values
    indices = indices.groupby(['x', 'y']).median().reset_index()
    
    x_size = int((LONDON_LONS[1] - LONDON_LONS[0])/lon_res) + 1    
    y_size = int((LONDON_LATS[1] - LONDON_LATS[0])/lat_res) + 1
    
    x_mask = (indices['x'] < 0) | (indices['x'] >= x_size)
    y_mask = (indices['y'] < 0) | (indices['y'] >= y_size)
    mask = ~(x_mask | y_mask)
    
    arr = sp.nan*sp.zeros((x_size, y_size))
    arr[indices.loc[mask, 'x'].values, indices.loc[mask, 'y'].values] = indices.loc[mask, 'values'].values
    
    return arr
    
def smooth(arr, sigma=5):
    filled = arr.copy()
    nans = sp.isnan(arr)    
    
    filled[nans] = 0
    smoothed = sp.ndimage.gaussian_filter(filled, sigma=sigma)  
    missing_coefs = sp.ndimage.gaussian_filter((~nans).astype(float), sigma=sigma)
    
    corrected = smoothed/missing_coefs
    corrected[nans] = sp.nan    
    
    return corrected
    
def show(arr, **kwargs):
    params = {
        'vmin': sp.nanpercentile(arr, 1),
        'vmax': sp.nanpercentile(arr, 99),
        'cmap': plt.cm.viridis,
        'interpolation': 'nearest'
    }    
    
    plt.imshow(arr.T[::-1], **dict(params, **kwargs))
    plt.colorbar(fraction=0.03)
    
def with_walking(time_arr, mins_per_square=1.3, constant=5):
    arr = time_arr.copy()
    cross_footprint = sp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(bool)
    diag_footprint = sp.array([[1, 0, 1],[0, 1, 0], [1, 0, 1]]).astype(bool)
    arr[sp.isnan(arr)] = sp.inf
    for i in range(60):
        cross_arr = sp.ndimage.minimum_filter(arr, footprint=cross_footprint)
        cross_arr[sp.isnan(cross_arr)] = sp.inf
        cross_changes = (cross_arr != arr)
        cross_arr[cross_changes] += 1*mins_per_square
    
        diag_arr = sp.ndimage.minimum_filter(arr, footprint=diag_footprint)
        diag_arr[sp.isnan(diag_arr)] = sp.inf
        diag_changes = (diag_arr != arr)
        diag_arr[diag_changes] += 1.4*mins_per_square
    
        arr = sp.minimum(cross_arr, diag_arr)
    
    arr[sp.isinf(arr)] = sp.nan
    
    return arr + constant
    
def get_relative_prices(walking_time, smoothed_prices):
    x = walking_time.flatten()
    y = smoothed_prices.flatten()
    mask = sp.isnan(x) | sp.isnan(y)
    
    spline = sp.interpolate.UnivariateSpline(x[~mask], y[~mask], s=len(x))
    v = spline(x)
    
    rel = (y - v).reshape(walking_time.shape)
    
    return rel
    
def fill_nans(arr, sigma=1):
    filled = arr.copy()
    nans = sp.isnan(arr)    
    
    filled[nans] = 0
    smoothed = sp.ndimage.gaussian_filter(filled, sigma=sigma, truncate=20)  
    missing_coefs = sp.ndimage.gaussian_filter((~nans).astype(float), sigma=sigma, truncate=20)
    
    corrected = smoothed/missing_coefs
    
    return corrected

def overlay_with_map(relative_prices):
    map_image = sp.ndimage.imread('map.png')
    filled_prices = fill_nans(sp.exp(relative_prices), 3)
    zoomed = sp.ndimage.zoom(filled_prices, map_image.shape[1]/float(filled_prices.shape[0]))
    
    plt.imshow(map_image.mean(2), interpolation='nearest', cmap=plt.cm.gray)
    plt.imshow(zoomed.T[::-1], alpha=0.5, interpolation='nearest', cmap=plt.cm.viridis, vmax=1.25, vmin=0.75)
    plt.gcf().set_size_inches(36, 36)
    plt.colorbar(fraction=0.03)
    plt.savefig('relative_prices.png', bbox_inches='tight')

def run():
    edges, locations = tfl.run()
    
    prices = get_price_paid_coords()
    price_arr = get_array(prices[['latitude', 'longitude']], prices.price)
    smoothed_prices = smooth(sp.log10(price_arr), 2)
    
    times = get_time_to_location(edges, locations)
    time_arr = get_array(times[['latitude', 'longitude']], times.time)
    walking_time = with_walking(time_arr)
    
    relative_prices = get_relative_prices(walking_time, smoothed_prices)
    
    overlay_with_map(relative_prices)
