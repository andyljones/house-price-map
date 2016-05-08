# -*- coding: utf-8 -*-
"""
Created on Sat May  7 08:55:29 2016

@author: andyjones
"""

import scipy as sp
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

LONDON_LATS = (51.25, 51.75)
LONDON_LONS = (-.5, +.5)

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
    
def get_price_paid_coords():
    postcodes = load_postcodes()
    price_paid = load_price_paid()[['price', 'date', 'postcode']]
    merged = pd.merge(price_paid, postcodes, left_on='postcode', right_index=True)
    return merged.drop('postcode', 1)
    
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
    
def filter_to_london(coords):
    lat_mask = (coords.latitude > LONDON_LATS[0]) & (coords.latitude < LONDON_LATS[1])    
    lon_mask = (coords.longitude > LONDON_LONS[0]) & (coords.longitude < LONDON_LONS[1])
    
    return coords[lat_mask & lon_mask]

#stations = pd.Series(times.index & locations.index)
#x = locations.longitude.loc[stations].values
#y = locations.latitude.loc[stations].values
#z = times.loc[stations].values
#
#f = sp.interpolate.NearestNDInterpolator(sp.vstack([x, y]).T, z)
#
#ny = prices.latitude.values
#nx = prices.longitude.values
#house_times = f(sp.vstack([nx, ny]).T)

#a = []
#q25 = []
#q50 = []
#q75 = []
#for t in sp.arange(10, 81):
#    window = ((house_times > t - 7) & (house_times < t + 7))
#    a.append(t)
#    q25.append(sp.percentile(prices.price[window], 25))
#    q50.append(sp.percentile(prices.price[window], 50))
#    q75.append(sp.percentile(prices.price[window], 75))
#
#x_noise = sp.random.normal(scale=1, size=house_times.shape)
#y_noise = sp.random.normal(scale=10000, size=prices.price.shape)
#plt.scatter(house_times + x_noise, prices.price + y_noise, marker='.', alpha=0.05, s=20, edgecolor='face', c='k')
#plt.ylim(0, 2e6)
#plt.xlim(10, 80)
#plt.fill_between(a, q25, q75, alpha=0.3)
#plt.plot(a, q25, a, q75, c=sns.color_palette()[0])
#plt.plot(a, q50, c=sns.color_palette()[0], linewidth=5)
#plt.gcf().set_size_inches(10, 8)