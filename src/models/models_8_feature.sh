#!/bin/bash
#
# bash script to run 8 feature trained models sequentially

python neural_network_5.py
python v2_random_forest.py
python lr_8_features.py
python knn_8_features.py
python gaussian_model.py
