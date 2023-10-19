import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pycountry_convert as pc
from typing import Tuple
#import panda as pd


def get_continent_name(continent_code: str) -> str:
    continent_dict = {
        "NA": "North America",
        "SA": "South America",
        "AS": "Asia",
        "AF": "Africa",
        "OC": "Oceania",
        "EU": "Europe",
        "AQ" : "Antarctica"
    }
    return continent_dict[continent_code]


def get_continent(lat: float, lon:float) -> Tuple[str, str]:
    geolocator = Nominatim(user_agent="<username>@gmail.com", timeout=10)
    geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    location = geocode(f"{lat}, {lon}", language="en")

    # for cases where the location is not found, coordinates are antarctica
    if location is None:
        return "Antarctica", "Antarctica"

    # extract country code
    address = location.raw["address"]
    country_code = address["country_code"].upper()

    # get continent code from country code
    continent_code = pc.country_alpha2_to_continent_code(country_code)
    continent_name = get_continent_name(continent_code)
    
    return continent_name


data = np.load('species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 


data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 

_, species_counts = np.unique(train_ids, return_counts=True)
species_count = np.bincount(train_ids)

top_sp_list = []
i = 0

for n in species_count:
    if n == species_counts.max():
        top_sp_list.append(i)
    i = i + 1

train_inds_pos = []
geometry = []

#Code copied from plot_world_map 
#j = 0
for n in top_sp_list:
    train_inds_pos.append(np.where(train_ids == n)[0])
    #geometry.append([Point(xy) for xy in zip(train_locs[train_inds_pos[j], 1], train_locs[train_inds_pos[j], 0])]) # gets list of (lat,lon) pairs
    #j += 1

#print(train_locs[train_inds_pos[0][0]]) 
#This is the first lat/lon of the first of the "most common species"

#HAVE TO CONVERT THESE LAT/LON INTO CONTINENTS AND THEN CREATE A DATA FRAME SUCH THAT FOR EACH OF THE TRAIN_LOCS
# THERE IS AN ASSOCIATED CONTINENT

latitude = train_locs[train_inds_pos[0][0]][0]
longitude = train_locs[train_inds_pos[0][0]][1]
continent = get_continent(latitude, longitude)

###### FOR GETTING THE CONTINENT OF 1 DATA POINT THIS WORKED, NOW HAVE TO GET 2000*50 OF THEM, NOT SURE IF IT WILL
# BE HAPPY ABOUT THAT... 

continents = []
k = 0
while k < len(train_inds_pos[0]):
    latitude = train_locs[train_inds_pos[0][k]][0]
    longitude = train_locs[train_inds_pos[0][k]][1]
    continent = get_continent(latitude, longitude)
    continents.append([latitude, longitude, continent])
    k += 1
    if k > 20:
        break

#Got the continent for the first 20 datapoints of the first most common species, all of them in North America for
#now which makes sense. I am guessing expanding this to 2000 points and 50 species will be problematic
# Also having trouble with panda for some reason??

#continents_df = pd.DataFrame(data, columns=['Latitude', 'Longitude', 'Continent'])

print(continents)
