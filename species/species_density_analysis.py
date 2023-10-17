# most densely populated species
# idea here is population(count of data points per species)/area(pi*r^2, where r = average distance per point) to give most and least
# dense populations --> data may be more suitable for a bar graph or something else idk

# also, not sure the units for the density as longitude and latitude is angular --> will have to convert

import numpy as np
import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt
import random
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point

def distance(x1, y1, x2, y2):
    return ((x2-x1)**2 + (y2-y1)**2)**(0.5)

# load train data
data = np.load('species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

# loading test data 
data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))   

species_train_dense = Path("species_train_density.csv")

if species_train_dense.is_file() == False: # need to create the data

    species_dense = []
    count = 0

    for sp in species: # for loop to estimate average distance between datapoints for a species
        test_inds_pos = test_pos_inds[sp]  
        dist = []
        # note: some species have a lot more data than others!
        # hence: going to artificially select a maximum of 500 data points to do this testing. imo should be enough
        if len(test_inds_pos)>500:
            test_inds_pos = random.sample(test_inds_pos, 500)
        for x in test_inds_pos:
            for y in test_inds_pos:
                if x != y:
                    val = distance(test_locs[x, 1], test_locs[x, 0], test_locs[y, 1], test_locs[y, 0])
                    dist.append(val)

        species_dense.append([sum(dist)/len(dist), sp])
        count += 1
        print(f"species {species_names[sp]} done!, average distribution: {sum(dist)/len(dist)}, {count} out of 500")

    with open('species_train_density.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(species_dense) # saves in file if file didn't exist

# now can analyze generated data

with open('species_train_density.csv', newline='') as f:
    reader = csv.reader(f)
    species_dense = list(reader)

density = []

for x in species_dense:
    area = math.pi*(float(x[0])**2)
    count = len(test_pos_inds[int(x[1])]) 

    density.append([count/area, int(x[1])])

sorted_species_density = sorted(density, key=lambda x: float(x[0]))

label_mini = int(sorted_species_density[0][1])
label_maxi = int(sorted_species_density[-1][1])

print(f"The most densely populated species is: {species_names[label_maxi]}") # finds label and species with highest count per area
print(f"The least densely populated species is: {species_names[label_mini]}") # ''' smallest '''


test_inds_pos_maxi = test_pos_inds[label_maxi]  

test_inds_pos_mini = test_pos_inds[label_mini]

# geopandas code to plot data

geometry_maxi = [Point(xy) for xy in zip(test_locs[test_inds_pos_maxi, 1], test_locs[test_inds_pos_maxi, 0])] 
geometry_mini = [Point(xy) for xy in zip(test_locs[test_inds_pos_mini, 1], test_locs[test_inds_pos_mini, 0])]

gdf_maxi = GeoDataFrame(geometry=geometry_maxi) 
gdf_mini = GeoDataFrame(geometry=geometry_mini)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
ax = world.plot(figsize=(10, 6))
gdf_maxi.plot(ax=ax, marker='o', color='b', markersize=5, label=f"{species_names[label_maxi]} ({label_maxi}), density: {float(sorted_species_density[-1][0])}")
gdf_mini.plot(ax=ax, marker='o', color='r', markersize=5, label=f"{species_names[label_mini]} ({label_mini}), density: {float(sorted_species_density[0][0])}")

plt.legend()
plt.title(f"Population distribution of most vs. least densely populated species.")
plt.show()
