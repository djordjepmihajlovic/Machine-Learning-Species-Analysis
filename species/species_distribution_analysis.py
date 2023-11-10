# for density run in terminal as: python species_distribution_analysis.py -d density
# change arg as needed i.e. -d distribution

import geopy.distance
import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import random
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import math
from argparse import ArgumentParser
import pandas as pd
import seaborn as sns
from pandas.plotting import table

def distance(x1, y1, x2, y2): 
    coords_1 = (x1, y1)
    coords_2 = (x2, y2)
    dist = geopy.distance.geodesic(coords_1, coords_2).km

    return dist

def get_params(): # runs chosen problem
    par = ArgumentParser()
    par.add_argument("-d", "--problem", type=str, default="distribution", help="Options: distribution, density, ...")
    args = par.parse_args()

    return args

def main():
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

    # loading train_extra data
    data_extra = np.load('species_train_extra.npz',) 

    species_train_dist = Path("species_train_dist.csv")
    species_train_dense = Path("species_train_dense.csv")

    if species_train_dist.is_file() == False or species_train_dense.is_file() == False: # need to create the data

        species_distn = []
        species_dense = []
        count = 0

        for sp in species: # for loop to estimate largest distance between datapoints for a species
            test_inds_pos = test_pos_inds[sp]  
            dist = []
            # note: some species have a lot more data than others!
            # hence: going to artificially select a maximum of 50 data points to do this testing. imo should be enough
            if len(test_inds_pos)>50:
                test_inds_pos = random.sample(test_inds_pos, 50)
            for x in test_inds_pos:
                for y in test_inds_pos:
                    if x != y:
                        # nb. [x coord. latitude(0), y coord. longitude(1)]
                        val = distance(test_locs[x, 0], test_locs[x, 1], test_locs[y, 0], test_locs[y, 1])
                        dist.append(val)

            species_distn.append([max(dist), sp])
            species_dense.append([sum(dist)/len(dist), sp])
            count += 1
            print(f"species {species_names[sp]} done!, max distribution: {max(dist)}, {count} out of 500")

        with open('species_train_dist.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(species_distn) # saves in file if file didn't exist - now function doesn't need to be run

        with open('species_train_dense.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(species_dense) # saves in file if file didn't exist - now function doesn't need to be run

    # now can analyze generated data

    if d == "distribution":

        with open('species_train_dist.csv', newline='') as f:
            reader = csv.reader(f)
            species_distn = list(reader)

        sorted_species_distn = sorted(species_distn, key=lambda x: float(x[0]))

        label_mini = int(sorted_species_distn[0][1])
        label_maxi = int(sorted_species_distn[-1][1])

        print(f"The largest distributed species is: {species_names[label_maxi]}, spanning: {float(sorted_species_distn[-1][0])}km") # finds label and species with largest spread
        print(f"The smallest distributed species is: {species_names[label_mini]}, spanning: {float(sorted_species_distn[0][0])}km ") # ''' smallest '''


        test_inds_pos_maxi = test_pos_inds[label_maxi]  

        test_inds_pos_mini = test_pos_inds[label_mini]

        # geopandas code to plot data

        geometry_maxi = [Point(xy) for xy in zip(test_locs[test_inds_pos_maxi, 1], test_locs[test_inds_pos_maxi, 0])] 
        geometry_mini = [Point(xy) for xy in zip(test_locs[test_inds_pos_mini, 1], test_locs[test_inds_pos_mini, 0])]

        gdf_maxi = GeoDataFrame(geometry=geometry_maxi) 
        gdf_mini = GeoDataFrame(geometry=geometry_mini)

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
        ax = world.plot(figsize=(10, 6))
        gdf_maxi.plot(ax=ax, marker='o', color='b', markersize=5, label=f"{species_names[label_maxi]} span: {float(sorted_species_distn[-1][0])} km")
        gdf_mini.plot(ax=ax, marker='o', color='r', markersize=5, label=f"{species_names[label_mini]} span: {float(sorted_species_distn[0][0])} km")

        plt.legend()
        plt.title(f"Population distribution of most localized vs. most spread species.")
        plt.show()

    elif d == "density":

        with open('species_train_dense.csv', newline='') as f:
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

        print(len(test_pos_inds[int(label_maxi)]))

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
        gdf_maxi.plot(ax=ax, marker='o', color='b', markersize=5, label=f"{species_names[label_maxi]} ({label_maxi}), occurence: {round(1/float(sorted_species_density[-1][0]), 2)} km²")
        gdf_mini.plot(ax=ax, marker='o', color='r', markersize=5, label=f"{species_names[label_mini]} ({label_mini}), occurence: {round(1/float(sorted_species_density[0][0]), 2)} km²")

        plt.legend()
        plt.title(f"Population distribution of most vs. least densely populated species.")
        plt.show()

    elif d == "heatmap":

        positive_index = data_test['test_pos_inds']
        present = []

        for idx, i in enumerate(positive_index):
            for j in i:
                if j not in present:
                    present.append(j)
            print(idx)

        test_locs_with_species = [j for i, j in enumerate(test_locs) if i in present]

        df = pd.DataFrame(test_locs_with_species, columns=["Latitude", "Longitude"])

        mean_latitude = df['Latitude'].mean()
        mean_longitude = df['Longitude'].mean()

        joint_plot = sns.jointplot(data = df, x= 'Longitude', y = 'Latitude', kind = 'hist', marginal_kws=dict(bins=20), bins=20)
        joint_plot.ax_joint.scatter(mean_longitude, mean_latitude, color='red', marker='X', label='Mean')

        joint_plot.ax_joint.legend()
        plt.savefig('/Users/djordjemihajlovic/Desktop/Theoretical Physics/Group Projects/AML Group Project/AML_GP/species/hist_test.png')
        plt.show()

    elif d == "describe":

        df_train = pd.DataFrame(train_locs, columns=["Latitude", "Longitude"])
        df_test = pd.DataFrame(test_locs, columns=["Latitude", "Longitude"])
        describe_train = df_train.describe()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        table(ax, describe_train, loc='center', colWidths=[0.2, 0.2, 0.2, 0.2])

        plt.show()

    elif d == "species":

        species_diff = []
        print(data_test["taxon_ids"])
        print(data["taxon_ids"])

        for i in data_test["taxon_ids"]:
            if i not in data["taxon_ids"]:
                species_diff.append(i)

        if species_diff == []:
            print("Species set is the same!")

        x = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
        y = [4, 5, 6, 6, 4 ,2, 1, 6, 9]

        sns.barplot(x = x, y = y)
        plt.xlabel("metric")
        plt.ylabel("accuracy")
        plt.show()

    elif d == "extra":

        # just for seeing what is in species_train_extra - its same as species_train just with new species (not in test)
        # also testing climate data 
        # climate date format is like so:
        # I is a 1080x2160 image (map) at each pixel there is a value indicating what i think is gray scale color [r, g, b, 255]? indicating variability
        # there are 19 images

        # 1 = annual mean temp.     
        # 2 = mean change in temp.     
        # 3 = change in day-night temp. 
        # 4 = temp std.  
        # 5 = max temp.  
        # 6 = min temp.
        # 7 = temp range.
        # ...

        I = plt.imread('wc2/wc2.1_10m_bio_4.tif')
        print(len(I), len(I[0]), len(I[0][0])) # y, x, val
        x = []
        y = []
        for i in range(0, 1080):
            print(I[i][1500])
            if I[i][1500][2] != 0: # fix x at 1080 (vertical line down middle of map)
                y.append(i)
                x.append(1500)

        plt.scatter(x, y)
        plt.xlim([0, 2160])
        plt.ylim([0, 1080])
        plt.show()






if __name__ == "__main__":
    args = get_params()
    d = args.problem

    main()







    