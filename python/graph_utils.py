'''
http://stackoverflow.com/questions/9651940/determining-and-storing-voronoi-cell-adjacency
'''
import numpy as np

import pyvoro

import pickle
import logging
import copy

from collections import defaultdict,Counter
import itertools

def build_adjacent_set(locations,distance_percentage=0.1):
    n,d = locations.shape
    
    if d == 2:
        return build_adjacent_set_2d(locations,distance_percentage=distance_percentage)
    elif d == 3:
        return build_adjacent_set_3d(locations,distance_percentage=distance_percentage)
    else:
        raise Exception("wrong location dismensions")

def build_adjacent_set_2d(locations,distance_percentage=0.1):
    n,m = locations.shape
    
    min_locations = np.min(locations,axis=0)
    max_locations = np.max(locations,axis=0)
    
    range_locations = max_locations - min_locations
    
    boundary_distance = range_locations * distance_percentage

    grid_box = [
            [min_locations[0]-boundary_distance[0],max_locations[0]+boundary_distance[0]],
            [min_locations[1]-boundary_distance[1],max_locations[1]+boundary_distance[1]]
        ]


    cells = pyvoro.compute_2d_voronoi(locations,grid_box,1.0)

    neiList = defaultdict(set)

    for i in xrange(n):
        for face in cells[i]["faces"]:
            ajd_cell = face["adjacent_cell"]
            if ajd_cell >= 0:
                neiList[i].add(ajd_cell)

        

    return (neiList, cells)


def build_adjacent_set_3d(locations,distance_percentage=0.1):
    n,m = locations.shape
    
    min_locations = np.min(locations,axis=0)
    max_locations = np.max(locations,axis=0)
    
    range_locations = max_locations - min_locations
    
    boundary_distance = range_locations * distance_percentage

    grid_box = [
            [min_locations[0]-boundary_distance[0],max_locations[0]+boundary_distance[0]],
            [min_locations[1]-boundary_distance[1],max_locations[1]+boundary_distance[1]],
            [min_locations[2]-boundary_distance[2],max_locations[2]+boundary_distance[2]]
        ]


    cells = pyvoro.compute_voronoi(locations,grid_box,1.0)

    neiList = defaultdict(set)

    for i in xrange(n):
        for face in cells[i]["faces"]:
            ajd_cell = face["adjacent_cell"]
            if ajd_cell >= 0:
                neiList[i].add(ajd_cell)

        

    return (neiList, cells)
