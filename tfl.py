# -*- coding: utf-8 -*-
"""
Created on Sun May  8 11:19:10 2016

@author: andyjones
"""

import scipy as sp
import time
import json
import pandas as pd
import requests
import os
import bs4
from fastkml.kml import KML

ROOT = 'https://api.tfl.gov.uk/'
PARAMS = {
    'app_id': '64dd8f8c',
    'app_key': '8f945b259bf29848031c8dbdb7abf4b3',
}

def load_train_stations():
    k = KML()
    k.from_string(open('data/stations.kml').read())
    features = list(k.features())[0].features()
    features = [[f.name.strip(), f.geometry.x, f.geometry.y] for f in features]
    features = pd.DataFrame(features, columns=['name', 'x', 'y'])
    return features
    
def load_bus_stops():
    return pd.read_csv('data/bus-stops.csv')

def call_api(endpoint, **kwargs):
    result = requests.get(ROOT + endpoint, params=dict(PARAMS, **kwargs))
    return json.loads(result.content)
    
def get_routes():
    data = call_api('Line/Route')
    results = []
    for route in data:
        for section in route['routeSections']:
            results.append({
                'route_id': route['id'],
                'mode': route['modeName'],
                'route_name': route['name'],
                'destination_id': section['destination'],
                'destination_name': section['destinationName'],
                'origin_id': section['originator'],
                'origin_name': section['originationName'],
                'section_name': section['name'],
                'direction': section['direction']
            })
            
    return pd.DataFrame(results)
    
def get_route_info(line_id, direction):
    path = os.path.join('cache/routes', '{}-{}.json'.format(line_id, direction))  
    if not os.path.exists(path):
        print('Fetching {}-{}'.format(line_id, direction))
        data = call_api('Line/{}/Route/Sequence/{}'.format(line_id, direction))
        json.dump(data, open(path, 'w+'))
        
    return json.load(open(path))
            
def cache_route_info(routes):
    route_dirs = routes[['route_id', 'direction']].drop_duplicates()
    for i, (_, (route_id, direction)) in enumerate(route_dirs.iterrows()):
        print i
        get_route_info(route_id, direction)
        
def get_stop_sequences(routes):
    results = {}
    route_dirs = routes[['route_id', 'direction']].drop_duplicates()
    for _, (route_id, direction) in route_dirs.iterrows():
        route_info = get_route_info(route_id, direction)
        for i, sequence in enumerate(route_info['orderedLineRoutes']):
            results[(route_id, direction, i)] = sequence['naptanIds']
        
    return pd.Series(results)
    
def get_station_locations(routes):
    results = []
    route_dirs = routes[['route_id', 'direction']].drop_duplicates()
    for _, (route_id, direction) in route_dirs.iterrows():
        route_info = get_route_info(route_id, direction)
        for station in route_info['stations']:
            results.append({
                'id': station['id'],
                'name': station['name'],
                'latitude': station.get('lat', sp.nan),
                'longitude': station.get('lon', sp.nan)
            })
            
    results = pd.DataFrame(results).drop_duplicates('id').set_index('id')
    return results
    
def get_timetable(route_id, origin, destination):
    path = os.path.join('cache/timetables', '{}-{}-{}.json'.format(route_id, origin, destination))  
    if not os.path.exists(path):
        print('Fetching {}-{}-{}'.format(route_id, origin, destination))
        data = call_api('Line/{}/Timetable/{}/to/{}'.format(route_id, origin, destination))
        json.dump(data, open(path, 'w+'))
        
    return json.load(open(path))
    
def cache_timetables(routes):
    route_dirs = routes[['route_id', 'origin', 'destination']].drop_duplicates()
    for i, (_, (route_id, origin, destination)) in enumerate(route_dirs.iterrows()):
        print i
        get_route_info(route_id, origin, destination)