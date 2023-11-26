#!/bin/bash
#
# bash script to run 2 feature trained models sequentially

python neural_network.py
python random_forest.py
python log_regression_model.py
python knn_model.py
python gaussian_model.py
