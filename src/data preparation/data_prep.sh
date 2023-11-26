#!/bin/bash
#
# bash script to run data preparation tasks
# NB. all data from running these is already included and so running of these scripts is unneeded for models_*.sh to work

python species_distribution_analysis.py -d distribution
python species_distribution_analysis.py -d density
python species_distribution_analysis.py -d extra
python create_df_continent.py
python temperature_anomaly.py
python top_species_analysis.py