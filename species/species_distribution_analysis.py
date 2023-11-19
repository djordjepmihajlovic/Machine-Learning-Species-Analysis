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
import math

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

        # sns.set_style('darkgrid')

        with open('species_train_dist.csv', newline='') as f:
            reader = csv.reader(f)
            species_distn = list(reader)

        sorted_species_distn = sorted(species_distn, key=lambda x: float(x[0]))

        label_mini = sorted_species_distn[0:5]
        label_maxi = sorted_species_distn[-6:-1]


        print(label_mini)
        print(label_maxi)

        # print(f"The 5 largest distributed species are: {[int(i[1]) for i in label_maxi]}, spanning: {[float(j[0]) for j in label_maxi]}km") # finds label and species with largest spread
        # print(f"The 5 smallest distributed species are: {[int(i[1]) for i in label_mini]}, spanning: {[float(j[0]) for j in label_mini]}km ") # ''' smallest '''


        test_inds_pos_maxi = test_pos_inds[int(label_maxi[-1][1])]  

        test_inds_pos_mini = test_pos_inds[int(label_mini[0][1])]

        # geopandas code to plot data

        geometry_maxi = [Point(xy) for xy in zip(test_locs[test_inds_pos_maxi, 1], test_locs[test_inds_pos_maxi, 0])] 
        geometry_mini = [Point(xy) for xy in zip(test_locs[test_inds_pos_mini, 1], test_locs[test_inds_pos_mini, 0])]

        gdf_maxi = GeoDataFrame(geometry=geometry_maxi) 
        gdf_mini = GeoDataFrame(geometry=geometry_mini)

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
        ax = world.plot()
        gdf_maxi.plot(ax=ax, marker='o', color='b', markersize=5, label=f"{species_names[int(label_maxi[-1][1])]} span: {float(sorted_species_distn[-1][0])} km")
        gdf_mini.plot(ax=ax, marker='o', color='r', markersize=5, label=f"{species_names[int(label_mini[0][1])]} span: {float(sorted_species_distn[0][0])} km")

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

        label_mini = sorted_species_density[0]
        label_maxi = sorted_species_density[-2]


        print(label_mini)
        print(label_maxi)

        # print(f"The 5 densest distributed species are: {[int(i[1]) for i in label_maxi]}, spanning: {[float(j[0]) for j in label_maxi]}km") # finds label and species with largest spread
        # print(f"The 5 sparsest distributed species are: {[int(i[1]) for i in label_mini]}, spanning: {[float(j[0]) for j in label_mini]}km ") # ''' smallest '''


        test_inds_pos_maxi = test_pos_inds[int(label_maxi[1])]  

        test_inds_pos_mini = test_pos_inds[int(label_mini[1])]

        # geopandas code to plot data

        geometry_maxi = [Point(xy) for xy in zip(test_locs[test_inds_pos_maxi, 1], test_locs[test_inds_pos_maxi, 0])] 
        geometry_mini = [Point(xy) for xy in zip(test_locs[test_inds_pos_mini, 1], test_locs[test_inds_pos_mini, 0])]

        gdf_maxi = GeoDataFrame(geometry=geometry_maxi) 
        gdf_mini = GeoDataFrame(geometry=geometry_mini)

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
        ax = world.plot(figsize=(10, 6))
        gdf_maxi.plot(ax=ax, marker='o', color='b', markersize=5, label=f"{species_names[label_maxi[1]]} ({label_maxi[1]}), species count per km²: {round(float(label_maxi[0]), 5)}")
        gdf_mini.plot(ax=ax, marker='o', color='r', markersize=5, label=f"{species_names[label_mini[1]]} ({label_mini[1]}), species count per km²: {round(float(label_mini[0]), 5)}")

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

    elif d == "plot_result":
    
        top_dist_names = ['Podiceps' + '\n' + ' cristatus', 'Turdus' + '\n' + ' merula', 'Acanthis' + '\n' + ' flammea', 'Fregata' + '\n' + ' minor', 'Oceanites' + '\n' + ' oceanicus']
        top_ID = [4208, 12716, 145300, 4636, 4146]
        top_dist_data = [16079.319321173714, 17684.382767361738, 19166.129948662958, 19689.434592637197, 19833.606659853107]
        min_dist_names = ['Gallotia' + '\n' + 'stehlini', 'Paramesotriton' + '\n' + 'hongkongensis', 'Phoenicolacerta' + '\n' + 'troodica', 'Selasphorus' + '\n' + 'flammula', 'Rhyacotriton' + '\n' + 'kezeri']
        min_ID = [35990, 64387, 73903, 6364, 27696]
        min_dist_data = [48.66426342304116, 163.1234814187649, 191.2538733659659, 201.2622172061399, 237.33474103956044]

        top_dense_ID = [38992, 29976, 8076, 145310, 4569]
        top_dense_data = [0.0006965029544433499, 0.0007055364299438898, 0.0007106531885470116, 0.0007337961594351512, 0.0007513834632404488]
        top_sparse_ID = [4345, 44570, 42961, 32861, 2071]
        top_sparse_data = [4.177038786428789e-05, 4.221188087170338e-05, 4.477237034420791e-05, 4.7904293356840265e-05, 4.95147014244629e-05]

        sns.set_theme()
        df_top = pd.DataFrame({'Distn.s': top_dense_data}, index=top_dense_ID)
        df_min = pd.DataFrame({'Distn.s': top_sparse_data}, index=top_sparse_ID)

        df_combined = pd.concat([df_top, df_min], keys=['Densest', 'Sparsest'])

        ax = df_combined.unstack(level=0).plot(kind='bar', rot=0, legend=False)
        ax.set_xlabel('Species taxon ID')
        ax.set_ylabel('Avg. species count per km²')
        ax.set_title('Population density, top-5 densest vs top-5 sparsest.')
        plt.legend(title= 'Distribution type', loc='upper right', labels=['Densest', 'Sparsest'])
        plt.show()

        # ²

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

        bio1 = plt.imread('wc2/wc2.1_10m_bio_1.tif') # mean temp
        bio12 = plt.imread('wc2/wc2.1_10m_bio_12.tif') # precip
        bio_elev = plt.imread('wc2/wc2.1_10m_elev.tif') # elev
        bio4 = plt.imread('wc2/wc2.1_10m_bio_4.tif')
        bio15 = plt.imread('wc2/wc2.1_10m_bio_15.tif')

        # temperature related
        bio7 = plt.imread('wc2/wc2.1_10m_bio_7.tif') # mean temp range
        bio10 = plt.imread('wc2/wc2.1_10m_bio_10.tif') # mean temp (cold quarter)
        bio11 = plt.imread('wc2/wc2.1_10m_bio_11.tif') # mean temp (warm quarter)

        # precipitation related
        bio12 = plt.imread('wc2/wc2.1_10m_bio_12.tif') # annual precip
        bio14 = plt.imread('wc2/wc2.1_10m_bio_14.tif') # precip (driest month)
        bio15 = plt.imread('wc2/wc2.1_10m_bio_15.tif') # precip seasonality 




        x_len = len(bio1)
        y_len = len(bio1[0])
        x = []
        y = []
        heatmap_elev = np.zeros((x_len, y_len))
        heatmap_temp = np.zeros((x_len, y_len))
        heatmap_precip = np.zeros((x_len, y_len))
        heatmap_temp_seas = np.zeros((x_len, y_len))
        heatmap_precip_seas = np.zeros((x_len, y_len))

        # temp heatmaps
        heatmap_temp_range = np.zeros((x_len, y_len))
        heatmap_temp_cold = np.zeros((x_len, y_len))
        heatmap_temp_warm = np.zeros((x_len, y_len))

        # precip heatmaps
        heatmap_precip = np.zeros((x_len, y_len))
        heatmap_precip_dry = np.zeros((x_len, y_len))
        heatmap_precip_season = np.zeros((x_len, y_len))

        for j in range(0, 2160):  # conversion is 6...
            for i in range(0, 1080):
                heatmap_temp_range[i][j] = bio7[i][j][0]
                heatmap_temp_cold[i][j] = bio10[i][j][0]
                heatmap_temp_warm[i][j] = bio11[i][j][0]
                heatmap_precip[i][j] = bio12[i][j][0]
                heatmap_precip_dry[i][j] = bio14[i][j][0]
                heatmap_precip_season[i][j] = bio15[i][j][0]

        empt = []

        for idx, i in enumerate(test_locs): # lat lon
            latitude = i[0] # -90 -> 90
            lat_conv = -6*(latitude-90)
            longitude = i[1] # -180 -> 180
            long_conv = 6*(longitude+180)

            temp_range = heatmap_temp_range[int(lat_conv)][int(long_conv)]
            temp_cold = heatmap_temp_cold[int(lat_conv)][int(long_conv)]
            temp_warm = heatmap_temp_warm[int(lat_conv)][int(long_conv)]

            precip = heatmap_precip[int(lat_conv)][int(long_conv)]
            precip_dry = heatmap_precip_dry[int(lat_conv)][int(long_conv)]
            precip_season = heatmap_precip_season[int(lat_conv)][int(long_conv)]

            new_data = [i[0], i[1], temp_range, temp_cold, temp_warm, precip, precip_dry, precip_season]
            empt.append(new_data)

            print(idx)

        with open('species_test_8_features.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(empt) 

        k = -500

        idx = np.argpartition(heatmap_elev.ravel(), k)
        p = tuple(np.array(np.unravel_index(idx, heatmap_elev.shape))[:, range(min(k, 0), max(k, 0))])
        x_rank = [(i/6)-180 for i in p[1].tolist()]
        y_rank = [(-i/6)+90 for i in p[0].tolist()]

        locations = list(zip(y_rank, x_rank))

        elev_var = pd.DataFrame(heatmap_elev) # heatmaps
        precip_var = pd.DataFrame(heatmap_precip_seas)
        temp_var = pd.DataFrame(heatmap_temp)


    elif d == "extrac":
    
        data_clim = np.load('scores.npy')
        print(data_clim[0:10])




if __name__ == "__main__":
    args = get_params()
    d = args.problem

    main()







    