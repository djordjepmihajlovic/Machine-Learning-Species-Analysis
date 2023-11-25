"""
I want to create a dataframe which include the most prominent continent for all 500 species.
This is either not working or taking too long, dont get an error but I dont get a dataframe either
Adding continent as a 'feature' could be useful? Maybe? Even if the information isnt technically new.

Need to revise the code becuase i tried copying and pasting but it hasnt worked.. will do that soon
"""



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


# Read the CSV file into a DataFrame
df = pd.read_csv("../../data/all_species_continent_data.csv")

data = np.load('../../data/species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 
"""
_, species_counts = np.unique(train_ids, return_counts=True)
species_count = np.bincount(train_ids)

top500_sp_list = []

for id in species:
    index = np.where(train_ids == id)[0]
    count = len(index)
    #print(count)
    if count > 0 and count <= 500:
        top500_sp_list.append(id)


filter1 = df[df['Species ID'].isin(top500_sp_list)]






# Use value_counts() to get the count of rows per continent
continent_counts = filter1['Continent'].value_counts()

# Print or use the counts as needed
print(continent_counts)


"""

# Filter the DataFrame to get species IDs for entries where the continent is 'Europe'
europe_species_ids = df.loc[df['Continent'] == 'South America', 'Species ID'].tolist()

# Print or use the list of species IDs for Europe as needed
#print(europe_species_ids)
i = 0
for id in europe_species_ids:
    index = np.where(train_ids == id)
    #print(index)
    x = len(index[0])
    #print(x)
    i += x

print(i)



index = np.where(train_ids == 54549)
print(len(index[0]))


"""

Below the code to create the continent data 

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

def get_continent(lat: float, lon: float) -> Tuple[str, str]:
    geolocator = Nominatim(user_agent="<username>@gmail.com", timeout=10)
    geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    try:
        location = geocode(f"{lat}, {lon}", language="en")

        # for cases where the location is not found, coordinates ASSUME antarctica!!!!!!!!!!!!!!??? Could be sea!
        if location is None:
            return "Antarctica", "Antarctica"

        # extract country code
        address = location.raw["address"]
        country_code = address["country_code"].upper()

        if country_code == 'TL':
            continent_name = 'AS'
            return continent_name
        elif country_code:
            continent_code = pc.country_alpha2_to_continent_code(country_code)
            continent_name = get_continent_name(continent_code)
            return continent_name
        else:
            return "Antarctica", "Antarctica"

    except KeyError:
        # Handle the KeyError here (e.g., print a message or return a default value)
        print("KeyError: 'country_code' not found in address")
        return "Unknown", "Unknown"


data = np.load('species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

species_df = pd.DataFrame(columns=['Species ID', 'Species Name', 'Continent']) #Should I add the number of 


train_inds_pos = []
for n in species:
    train_inds_pos.append(np.where(train_ids == n)[0])

i = 0
for species_indices in train_inds_pos:
    # Randomly select 10 indices from each species
    train_inds_pos_sp = np.random.choice(species_indices, 10, replace=False)
    species_idf = species[i]
    species_namef = species_names[species[i]]
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

    #species_df = species_df.append({'Species ID': species_idf, 'Species Name': species_namef, 'Continent': most_common_continent}, ignore_index=True)
    species_df = pd.concat([species_df, pd.DataFrame([{'Species ID': species_idf, 'Species Name': species_namef, 'Continent': most_common_continent}])], ignore_index=True)


    print('Done species :', i)


# Save the species_df DataFrame to a CSV file
species_df.to_csv('all_species_continent_data.csv', index=False)

"""