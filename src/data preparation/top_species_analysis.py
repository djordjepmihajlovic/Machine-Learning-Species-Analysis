import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pycountry_convert as pc
from typing import Tuple
import random
import pandas as pd
from collections import Counter


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


data = np.load('../../data/species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 


data_test = np.load('../../data/species_test.npz', allow_pickle=True)
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

print(len(top_sp_list))
print(top_sp_list[32])
train_inds_pos = []


for n in top_sp_list:
    train_inds_pos.append(np.where(train_ids == n)[0])
    

print(len(train_inds_pos))
print(train_inds_pos[0])
print(train_inds_pos[1])

species_df = pd.DataFrame(columns=['Species ID', 'Species Name', 'Continent'])

i = 0
for species_indices in train_inds_pos: 
    # Randomly select 20 indices from each species
    train_inds_pos_sp = np.random.choice(species_indices, 20, replace=False)
    species_idf = top_sp_list[i]
    species_namef = species_names[top_sp_list[i]]
    i += 1
    continents = []
    # Loop through the selected indices for the current species
    for sample in train_inds_pos_sp:
        lat = train_locs[sample][0]
        lon = train_locs[sample][1]
        continent = get_continent(lat, lon)
        continents.append(continent)
    continents_count = Counter(continents)
    most_common_continent = continents_count.most_common(1)[0][0]

    species_df = species_df.append({'Species ID': species_idf, 'Species Name': species_namef, 'Continent': most_common_continent}, ignore_index=True)


# Save the species_df DataFrame to a CSV file
species_df.to_csv('species_continent_data.csv', index=False)


#species_df = pd.read_csv('species_continent_data.csv')

europe_species = species_df[species_df['Continent'] == 'Europe']['Species Name'].tolist()

print("Most common species in Europe are:")
for species in europe_species:
    print(species)

NA_species = species_df[species_df['Continent'] == 'North America']['Species Name'].tolist()

print("Most common species in North America are:")
for species in NA_species:
    print(species)

oceania_species = species_df[species_df['Continent'] == 'Oceania']['Species Name'].tolist()

print("Most common species in Oceania are:")
for species in oceania_species:
    print(species)


africa_species = species_df[species_df['Continent'] == 'Africa']['Species Name'].tolist()

print("Most common species in Africa are:")
for species in africa_species:
    print(species)


SA_species = species_df[species_df['Continent'] == 'South America']['Species Name'].tolist()

print("Most common species in South America are:")
for species in SA_species:
    print(species)

antarctica_species = species_df[species_df['Continent'] == 'Antarctica']['Species Name'].tolist()

print("Most common species in Antarctica are:")
for species in antarctica_species:
    print(species)

asia_species = species_df[species_df['Continent'] == 'Asia']['Species Name'].tolist()

print("Most common species in Asia are:")
for species in asia_species:
    print(species)







