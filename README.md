# LAMP: Learned Approximate Matrix Profile

This code provides an implementation of LAMP training and inference as described in "Time Series Mining in the Face of Fast Moving Streams using a Learned Approximate Matrix Profile"

# Usage:
~~~~
# For training + inference
python train_neural_net_LAMP.py <matrix profile window size> <input data path> <logging output path> <pretrained weights file (optional)> <initial epoch (optional)>
# For inference only
python train_neural_net_LAMP.py <matrix profile window size> <input data path> <logging output path> <pretrained weights file>
~~~~

# Data Format
Input data must be organized into a mat file with a specific set of variables defined: \ 
ts_train (time series input for training) \
mp_train (matrix profile output for training) \
ts_val (validation time series) \
mp_val (validation matrix profile) \
ts_test (testing time series) \
mp_test (testing matrix profile) \
