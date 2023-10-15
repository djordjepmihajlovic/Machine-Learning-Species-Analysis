import numpy as np
import csv
from geopy.geocoders import Nominatim
from pathlib import Path

data = np.load('species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

geolocator = Nominatim(user_agent="youremail@provider")

# generates countries associated to train data IF not already done as it takes long

test_country_path = Path("species_train_countries.csv")

if test_country_path.is_file() == False:
    train_country = []
    for i in range(0, len(train_locs)):
        location = geolocator.reverse(train_locs[i].tolist())
        train_country.append([i, location.raw["address"]["country"]])
    with open('species_train_countries.csv', 'w') as f:

        write = csv.writer(f)
        write.writerows(train_country)

    # now train_country provides a list corresponding w train_locs data points







    