# -*- coding: utf-8 -*-
"""
Created on Sun May  8 11:19:10 2016

@author: andyjones
"""
import networkx as nx
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
    
def get_timetable(route_id, origin, destination):
    path = os.path.join('cache/timetables', '{}-{}-{}.json'.format(route_id, origin, destination))  
    if not os.path.exists(path):
        print('Fetching {}-{}-{}'.format(route_id, origin, destination))
        data = call_api('Line/{}/Timetable/{}/to/{}'.format(route_id, origin, destination))
        json.dump(data, open(path, 'w+'))
        
    return json.load(open(path))
    
def walk_timetables(routes):    
    for _, row in routes.iterrows():
        data = get_timetable(row['route_id'], row['origin_id'], row['destination_id'])
        if ('timetable' in data) and (len(data['timetable']['routes']) > 0):
            yield data
        
def get_stops(route_id):
    path = os.path.join('cache/stoppoints', '{}.json'.format(route_id))  
    if not os.path.exists(path):
        print('Fetching {}'.format(route_id))
        data = call_api('Line/{}/StopPoints'.format(route_id))
        json.dump(data, open(path, 'w+'))
        
    return json.load(open(path))
    
def walk_stops(routes):
    for _, row in routes.iterrows():
        data = get_stops(row['route_id'])
        yield data    
        
def get_locations(routes):
    results = []
    for stops in walk_stops(routes):
        for stop in stops:
            results.append({
                'id': stop['id'],
                'naptan': stop['naptanId'],
                'station_naptan': stop.get('stationNaptan', ''),
                'hub_naptan': stop.get('hubNaptanCode', ''),
                'name': stop['commonName'],
                'latitude': stop['lat'],
                'longitude': stop['lon']
            })
            
    return pd.DataFrame(results).drop_duplicates('naptan').set_index('naptan')
    
def get_edges(routes):
    results = []
    for timetable in walk_timetables(routes):
        origin = timetable['timetable']['departureStopId']
        for route in timetable['timetable']['routes']:
            for intervals in route['stationIntervals']:
                stops = [origin] + [x['stopId'] for x in intervals['intervals']]
                edges = [[s, t] for s, t in zip(stops, stops[1:])]
                
                times = [0] + [x['timeToArrival'] for x in intervals['intervals']]
                weights = list(sp.diff(sp.array(times)))
                
                results.extend([[s, t, w] for (s, t), w in zip(edges, weights)])
    
    results = pd.DataFrame(results, columns=['origin', 'destination', 'time'])
    results = results.groupby(['origin', 'destination']).mean()
    
    return results

def get_travel_times(edges, locations, origin='940GZZLUGPK', transit_time=5):
    G = nx.Graph()
    G.add_weighted_edges_from(map(tuple, list(edges.reset_index().values)))
    
    for naptan, location in locations.iterrows():
        if location.hub_naptan != '':
            G.add_weighted_edges_from([(naptan, location.hub_naptan, transit_time)])
            
    times = nx.single_source_dijkstra_path_length(G, origin, weight='weight')
    return pd.Series(times)
    
def run():
    if not os.path.exists('edges.pkl'):
        routes = get_routes()
        get_edges(routes).to_pickle('edges.pkl')
        get_locations(routes).to_pickle('locations.pkl')
        
    return pd.read_pickle('edges.pkl'), pd.read_pickle('locations.pkl')
    
#to_green_park = nx.single_source_dijkstra_path_length(G, origin, weight='weight')
#points = locations.loc[to_green_park.keys(), ['latitude', 'longitude']].values
#colors = sp.array(to_green_park.values()).clip(0, 60)
#plt.scatter(points[:, 1], points[:, 0], c=colors, cmap=plt.cm.viridis_r, s=10, alpha=0.4, marker='.', edgecolor='face')
#plt.xlim(-0.5, 0.3)
#plt.ylim(51.3, 51.65)
#plt.colorbar()