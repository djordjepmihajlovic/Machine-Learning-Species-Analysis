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

    # for cases where the location is not found, coordinates ASSUME antarctica!!!!!!!!!!!!!!??? Could be sea!
    if location is None:
        return "Antarctica", "Antarctica"

    # extract country code
    address = location.raw["address"]
    country_code = address["country_code"].upper()

    # get continent code from country code
    #continent_code = pc.country_alpha2_to_continent_code(country_code) #Replaced these 3 lines for following code
    #continent_name = get_continent_name(continent_code)
    #return continent_name
    """
        try:
        continent_name = pc.country_alpha2_to_continent_code(country_code)

    except Exception as e:
        # Special Case: Timor Leste
        if e == "Invalid Country Alpha-2 code: \'TL\'":
            continent_name = 'AS'
        else:
            continent_name = 'N/A'
    
    if country_code:
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        try:
            continent_name = get_continent_name(continent_code)
        except Exception as e:
            if str(e) == '"Invalid Country Alpha-2 code: \'TL\'"':
                continent_name = 'AS'
            else:
                continent_name = 'N/A'
        return continent_name
    
    else:
        # Handle unrecognized country code, assuming it's Antarctica
        return "Antarctica", "Antarctica"
    """
    if country_code == 'TL':
        continent_name = 'AS'
        return continent_name
    elif country_code:
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        continent_name = get_continent_name(continent_code)
        return continent_name
    else:
        return "Antarctica", "Antarctica"


data = np.load('species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

species_df = pd.DataFrame(columns=['Species ID', 'Species Name', 'Continent']) #Should I add the number of 

#_, species_counts = np.unique(train_ids, return_counts=True)
#species_count = np.bincount(train_ids)

#all_sp_list = []
#i = 0

#for n in species_count:
#    if n < species_counts.max():
 #       all_sp_list.append(i)
  #  i = i + 1

#print(len(all_sp_list))

train_inds_pos = []
for n in species:
    train_inds_pos.append(np.where(train_ids == n)[0])

#print(len(train_inds_pos))
#print(train_inds_pos[0])
#print(train_inds_pos[1]) 

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

    #df = pd.DataFrame(df).append(new_row, ignore_index=True)
    #df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    #if i > :
    #    break

# Save the species_df DataFrame to a CSV file
species_df.to_csv('all_species_continent_data.csv', index=False)

