# Machine Learning for Species Distribution analysis 

### Python code for implementing and testing various models on an analysis of the distribution of species around the world

The following code is separated into three different folders; src, results and data. 

Here is a brief overview of the structure of the code layout:
* src: contains all code used to generate results from the project, separated into two different folders for implementing machine learning models and analysis/data preparation.
    * models: models contains all the models and implementations investigated
    * data preparation: code used to investigate data; such as finding sparse/dense populations, investigating top species and analyzing bioclimatic and climate data appropriately for training

* data: contains all data, given and generated, used to train and test models.
* results: found results and visualization methods (via seaborn and matplotlib)


## Running code and obtaining results

## src :

* /models:

    * Feed-forward neural network algorithms:

    To run the results for the Feed-Forward neural network trained on two features (latitude and longitude) run 
    'neural_network.py' in the terminal as:
    python neural_network.py
    a variable 'p' is included at the top of the script; change between p = "analyze" or p = "plot" to either run an analysis of the neural network model and print AUCROC, AUCPR, F2 and cohen kappa measures or to plot the predicted distribution of a species respectively.

    To run the results for the Feed-Forward neural network trained on eight features run 'neural_network_5.py' in the terminal as:
    python neural_network_5.py 
    similary to the 2 feature model a variable 'p' is included at the top of the script; change between p = "analyze" or p = "plot" to either run an analysis of the neural network model and print AUCROC, AUCPR, F2 and cohen kappa measures or to plot the predicted distribution of a species respectively.

    * Random Forest algorithms:

    To run the results for the Random Forest trained on 2 features, run random_forest.py. This will return all the results and evaluation for the random forest, as well as a plot for a given species.

    To run the results for the model trained on 8 features, run v2_random_forest.py. This will return the evaluation. Note that only F2 was added to the report.

    * Logistic regression algorithms:

    * K-Nearest Neighbour algorithms:

    * Gaussian algorithm:

* /data preparation:

    Note, all data has already been generated and saved into the /data folder, however instructions to run the files to generate the data are given below:

    * different species distribution types:

    To obtain the results on the different species distribution types (dense vs sparse, and largest vs smallest span respectively) run 'species_distribution_analysis.py' in the terminal as:
    python species_distribution_analysis.py -d density
    python species_distribution_analysis.py -d distribution
    The data is then saved into a .csv file

    * generate top species continent analysis:

    * generate 8 feature data:

    To generate the 8 feature data by extracting from the data available from WorldClim run 'species_distribution_analysis.py' as:
    python species_distribution_analysis.py -d extra
    The data is then saved into a .csv file

    * generate temperature anomaly scores:








