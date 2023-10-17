import numpy as np
import csv
from geopy.geocoders import Nominatim
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

species_train_dist = Path("species_train_dist.csv")

if species_train_dist.is_file() == False: # need to create the data

    species_distn = []
    count = 0

    for sp in species: # for loop to estimate largest distance between datapoints for a species
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

        species_distn.append([max(dist), sp])
        count += 1
        print(f"species {species_names[sp]} done!, max distribution: {max(dist)}, {count} out of 500")

    with open('species_train_dist.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(species_distn) # saves in file if file didn't exist

# now can analyze generated data

with open('species_train_dist.csv', newline='') as f:
    reader = csv.reader(f)
    species_distn = list(reader)

maxi = 0 # artificial initial values
mini = 100
for x in species_distn:
    if float(x[0])>float(maxi):
        maxi = x[0]
        label_maxi = int(x[1])
    elif float(x[0])<float(mini):
        mini = x[0]
        label_mini = int(x[1])

print(f"The largest distributed species is: {species_names[label_maxi]}") # finds label and species with largest spread
print(f"The smallest distributed species is: {species_names[label_mini]}") # ''' smallest '''


test_inds_pos_maxi = test_pos_inds[label_maxi]  

test_inds_pos_mini = test_pos_inds[label_mini]

# geopandas code to plot data

geometry_maxi = [Point(xy) for xy in zip(test_locs[test_inds_pos_maxi, 1], test_locs[test_inds_pos_maxi, 0])] 
geometry_mini = [Point(xy) for xy in zip(test_locs[test_inds_pos_mini, 1], test_locs[test_inds_pos_mini, 0])]

gdf_maxi = GeoDataFrame(geometry=geometry_maxi) 
gdf_mini = GeoDataFrame(geometry=geometry_mini)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
ax = world.plot(figsize=(10, 6))
gdf_maxi.plot(ax=ax, marker='o', color='b', markersize=5, label=f"{species_names[label_maxi]} ({label_maxi})")
gdf_mini.plot(ax=ax, marker='o', color='r', markersize=5, label=f"{species_names[label_mini]} ({label_mini})")

plt.legend()
plt.title(f"Population distribution of most sparse vs. most spread species.")
plt.show()





##### Below commented out is way to find countries of data

# get train locations and plot
# train_inds_pos = np.where(train_ids == sp)[0]
# plt.plot(train_locs[train_inds_pos, 1], train_locs[train_inds_pos, 0], 'rx', label='train')
# print(train_inds_pos)

# plt.title(str(sp) + ' - ' + species_names[sp])
# plt.grid(True)
# plt.xlim([-180, 180])
# plt.ylim([-90, 90])
# plt.ylabel('latitude')
# plt.xlabel('longitude')
# plt.legend()
# plt.show()

# geolocator = Nominatim(user_agent="youremail@provider")

# # generates countries associated to train data IF not already done as it takes long

# test_country_path = Path("species_train_countries.csv")

# if test_country_path.is_file() == False:
#     train_country = []
#     for i in range(0, len(train_locs)):
#         location = geolocator.reverse(train_locs[i].tolist())
#         train_country.append([i, location.raw["address"]["country"]])
#     with open('species_train_countries.csv', 'w') as f:

#         write = csv.writer(f)
#         write.writerows(train_country)

#     # now train_country provides a list corresponding w train_locs data points







    