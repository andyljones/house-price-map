# -*- coding: utf-8 -*-
"""
Created on Sat May  7 08:55:29 2016

@author: andyjones
"""
import os

import scipy as sp
import scipy.ndimage
import scipy.interpolate
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

sns.set_style('ticks')

import tfl

LONDON_LONS = (-.35, +.15)
LONDON_LATS = (51.4, 51.65)

#TODO: Redo the price interpolation properly rather than binning it first.

PRICE_PAID_PATH = 'data/pp-2015.csv'
POSTCODES_PATH = 'data/ukpostcodes.zip'

OUTPUT_PATH = 'output'

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
    data = pd.read_csv(PRICE_PAID_PATH, header=None)
    data = data.rename(columns=PRICE_PAID_COLUMNS)
    return data
    
def load_postcodes():
    data = pd.read_csv(POSTCODES_PATH)
    return data.set_index('postcode')[['latitude', 'longitude']]
    
def filter_to_london(coords):
    lat_mask = (coords.latitude > LONDON_LATS[0]) & (coords.latitude < LONDON_LATS[1])    
    lon_mask = (coords.longitude > LONDON_LONS[0]) & (coords.longitude < LONDON_LONS[1])
    
    return coords[lat_mask & lon_mask]    
    
def get_price_paid_coords():
    postcodes = load_postcodes()
    price_paid = load_price_paid()[['price', 'date', 'postcode']]
    merged = pd.merge(price_paid, postcodes, left_on='postcode', right_index=True)
    
    london = filter_to_london(merged).copy()
    london['price'] = sp.log10(london['price'])
    return london.drop('postcode', 1)

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
    indices = indices.groupby(['x', 'y'])['values'].agg(['median', 'count']).reset_index()
    
    x_size = int((LONDON_LONS[1] - LONDON_LONS[0])/lon_res) + 1    
    y_size = int((LONDON_LATS[1] - LONDON_LATS[0])/lat_res) + 1
    
    x_mask = (indices['x'] < 0) | (indices['x'] >= x_size)
    y_mask = (indices['y'] < 0) | (indices['y'] >= y_size)
    mask = ~(x_mask | y_mask)
    indices = indices.loc[mask]
    
    arr = sp.nan*sp.zeros((x_size, y_size))
    arr[indices['x'].values, indices['y'].values] = indices['median'].values
    
    count_arr = sp.zeros((x_size, y_size)) 
    count_arr[indices['x'].values, indices['y'].values] = indices['count'].values
    
    return arr, count_arr
    
def smooth(arr, sigma=5):
    filled = arr.copy()
    nans = sp.isnan(arr)    
    
    filled[nans] = 0
    smoothed = sp.ndimage.gaussian_filter(filled, sigma=sigma, truncate=10)  
    missing_coefs = sp.ndimage.gaussian_filter((~nans).astype(float), sigma=sigma, truncate=10)
    
    corrected = smoothed/missing_coefs
    
    return corrected
    
def with_walking(time_arr, mins_per_square=1.3, transfer_constant=5):
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
    
    return arr + transfer_constant
    
def get_relative_prices(walking_time, smoothed_prices):
    x = walking_time.flatten()
    y = smoothed_prices.flatten()
    mask = sp.isnan(x) | sp.isnan(y)
    
    spline = sp.interpolate.UnivariateSpline(x[~mask], y[~mask], s=len(x))
    v = spline(x)
    
    rel = (y - v).reshape(walking_time.shape)
    
    return rel

def plot_over_map(arr, **kwargs):
    map_image = sp.ndimage.imread('map.png')
    filled_prices = smooth(arr, 1)
    zoomed = sp.ndimage.zoom(filled_prices, map_image.shape[1]/float(filled_prices.shape[0]))
    
    plt.imshow(map_image.mean(2), interpolation='nearest', cmap=plt.cm.gray)
    plt.imshow(zoomed.T[::-1], alpha=0.5, interpolation='nearest', cmap=plt.cm.viridis, **kwargs)
    
    plt.xticks([])
    plt.yticks([])

def plot_price(smoothed_prices):
    plot_over_map(10**(smoothed_prices - 3), norm=LogNorm(1.5e2, 1e3))
    cb = plt.colorbar(fraction=0.03, ticks=sp.linspace(2e2, 1e3, 9), format=FormatStrFormatter(u'£%dk'))
    cb.set_label(u'price paid (£1000s)')    
    
    plt.title('2015 Average Price Paid')
    plt.gcf().set_size_inches(36, 36)
    plt.gcf().savefig(os.path.join(OUTPUT_PATH, 'price_paid.png'), bbox_inches='tight')
    
def plot_time(walking_time):
    plot_over_map(walking_time, vmin=15, vmax=75)
    cb = plt.colorbar(fraction=0.03, ticks=sp.linspace(15, 75, 5))
    cb.set_label('commute time (mins)')    
    
    plt.title('Commute time to Green Park')
    plt.gcf().set_size_inches(36, 36)
    plt.gcf().savefig(os.path.join(OUTPUT_PATH, 'travel_time.png'), bbox_inches='tight')
    
def plot_relative_price(relative_prices):
    plot_over_map(10**relative_prices, norm=LogNorm(0.5, 2))
    cb = plt.colorbar(fraction=0.03, ticks=sp.linspace(0.5, 2, 4), format=FormatStrFormatter('x%.2f'))
    cb.set_label('fraction of average price paid for commute time')
    
    plt.title('Price relative to commute')
    plt.gcf().set_size_inches(36, 36)
    plt.gcf().savefig(os.path.join(OUTPUT_PATH, 'relative_price.png'), bbox_inches='tight')
    

def plot_price_with_time_contours(smoothed_prices, walking_time):
    map_image = sp.ndimage.imread('map.png')
    plt.imshow(map_image.mean(2), interpolation='nearest', cmap=plt.cm.gray)
    
    zoomed_prices = sp.ndimage.zoom(smoothed_prices, map_image.shape[1]/float(smoothed_prices.shape[0]))
    
    plt.imshow(zoomed_prices.T[::-1], alpha=0.5, interpolation='nearest', cmap=plt.cm.viridis, vmin=5.25, vmax=5.75)
    
    plt.colorbar(fraction=0.03)
    
    smoothed_times = smooth(walking_time, sigma=2)
    zoomed_times = sp.ndimage.zoom(smoothed_times, map_image.shape[1]/float(smoothed_prices.shape[0]))
    plt.contour(zoomed_times.T[::-1], cmap=plt.cm.Reds, levels=range(15, 61, 15), linewidths=3)
    
    plt.gcf().set_size_inches(36, 36)
    plt.savefig(os.path.join(OUTPUT_PATH, 'price_with_time_contours.png'), bbox_inches='tight')


def run():
    edges, locations = tfl.cache()
    
    prices = get_price_paid_coords()
    price_arr, count_arr = get_array(prices[['latitude', 'longitude']], prices.price)
    high_res_prices = smooth(price_arr, 2)
    low_res_prices = smooth(price_arr, 10)    
    weights = smooth(count_arr, 2).clip(0, 3)/3
    smoothed_prices = low_res_prices*(1-weights) + high_res_prices*(weights)
    
    times = get_time_to_location(edges, locations)
    time_arr, _ = get_array(times[['latitude', 'longitude']], times.time)
    walking_time = with_walking(time_arr)
    
    relative_prices = get_relative_prices(walking_time, smoothed_prices)
    
    plot_price(smoothed_prices)
    plt.clf()
    plot_time(walking_time)
    plt.clf()    
    plot_relative_price(relative_prices)
