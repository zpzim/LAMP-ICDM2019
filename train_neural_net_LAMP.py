import numpy as np
import math
import pandas as pd
import tensorflow as tf
import sys,os
from TimeSeriesGeneratorAdapt import MPTimeseriesGenerator
import keras
from scipy.stats import zscore
import scipy.io as sio
from keras.utils import multi_gpu_model
from tensorboard_callbacks import LRTensorBoard
from models import build_resnet
import datetime
import keras.backend as K

#Fix random seed so that we can reproduce results
np.random.seed(813306)
from tensorflow import set_random_seed
set_random_seed(5944)

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

if len(sys.argv) < 4:
  print('Usage: LAMP.py <mp_window_size> <root_data_dir> <root_logging_dir> <pretrained_weights_file (optional when training)> <initial_epoch (optional, default=0)>')
  exit(1)

# Flags indicating whether to train and/or predict
training = True
predicting = True

# If the training data is made up of multiple time series, then we need the endpoints of each individual sequence so that we don't generate input which overlaps multiple sequences
merge_points = None

# Variables which affect instance weighting based on the matrix profile value
high_weight = 1
low_thresh = -1
high_thresh = 1

# Variables corresponding to the matrix profile window size
matrix_profile_window = int(sys.argv[1])
# Input width should be the same as the matrix profile window
input_width = matrix_profile_window

# Variables affecting the kind of input we feed the neural network
# Every subsequence_stride datapoints in the input time series, we extract lookbehind + lookahead + num_outputs subsequences of length input_width
# Where lookbehind is the number of historical subsequences to extract, lookahead is the number of future subsequences to extract, and num_outputs
# is the number of outputs we predict for each input, the outputs are the matrix profile values for consecutive subsequences starting with the 'current'
# subsequence
sample_rate = 20
lookbehind_seconds = 0
lookahead_seconds = 0
subsequence_stride = 256
lookbehind = sample_rate * lookbehind_seconds
num_outputs = 256
lookahead = sample_rate * lookahead_seconds
forward_sequences = lookahead + num_outputs
subsequences_per_input = lookbehind + num_outputs + lookahead

# Channel stride is the stride between extracted subsequences in the input
channel_stride = 8

# Number of time series dimensions to consider
n_input_series= 1

subsequences_per_input = subsequences_per_input // channel_stride

# Neural network hyperparameters
initial_lr=1e-3
optimizer_id = 'Adam'
loss_function='mse'
batch_size = 32
nb_epochs = 30
init_epoch = 0

# Whether we shuffle the training data
shuffle = True

# Whether an RNN model is treated as stateful
stateful_rnn = False
optimizer = keras.optimizers.get(optimizer_id)
conf = optimizer.get_config()
conf['lr'] = initial_lr
conf['epsilon'] = 1e-4




root_data_path = sys.argv[2]
root_logging_path = sys.argv[3]

all_data = sio.loadmat(root_data_path)



if training:
  dataset_name = os.path.split(root_data_path)[-1]
  logging_filename = 'dataset={}_width={}_optimizer={}_initlr={}_batchsz={}_stride={}_shuffle={}_lookbehind={}_lookahead={}_channelstride={}_outputs={}_stateful={}_weight={}_lowthresh={}_highthresh={}_started={}'.format(dataset_name, input_width, optimizer_id, initial_lr, batch_size, subsequence_stride, shuffle, lookbehind, lookahead, channel_stride, num_outputs, stateful_rnn, high_weight, low_thresh, high_thresh, str(datetime.datetime.now()).replace(' ', '-'))
  model_path = os.path.join(root_logging_path, 'models', logging_filename)
  tensorboard_path = os.path.join(root_logging_path, 'tensorboard_logs', logging_filename)
  logging_path = os.path.join(root_logging_path, 'csv_logs', logging_filename)
  logging_dir = os.path.join(root_logging_path, 'csv_logs')
  # Make the directories if they don't already exists
  if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)

#If user provided an existing model, use that.
model_file = ''
if len(sys.argv) > 4:
  model_file = sys.argv[4]
  init_epoch = 0
  if len(sys.argv) > 5:
    init_epoch = int(sys.argv[5])

if model_file != '':
  model = keras.models.load_model(model_file)
else:
  # Build keras model
  with tf.device("/gpu:0"):
    #Change model as needed
    model = build_resnet([input_width, n_input_series, subsequences_per_input], 96, num_outputs)
  model.summary()
  # Compile model
  model.compile(loss=loss_function,
                optimizer=optimizer)
  model.summary()


if training:
  # Load training data
  mp = np.array(all_data['mp_train'])
  ts = np.array(all_data['ts_train'])

  # Load validation data
  mp_val = np.array(all_data['mp_val'])
  ts_val = np.array(all_data['ts_val'])

  # Construct generators for the model input
  train_gen =  MPTimeseriesGenerator(ts, mp, num_input_timeseries=n_input_series, internal_stride=channel_stride, num_outputs=num_outputs, lookahead=forward_sequences, lookbehind=lookbehind, length=input_width, mp_window=matrix_profile_window, stride=subsequence_stride, important_upper_threshold=high_thresh, important_lower_threshold=low_thresh, important_weight=high_weight, batch_size=batch_size, shuffle=shuffle, merge_points=merge_points)
  valid_gen =  MPTimeseriesGenerator(ts_val, mp_val, num_input_timeseries=n_input_series, internal_stride=channel_stride, num_outputs=num_outputs,lookahead=forward_sequences, lookbehind=lookbehind, important_upper_threshold=high_thresh, important_lower_threshold=low_thresh, important_weight=high_weight, length=input_width, mp_window=matrix_profile_window, stride=num_outputs, batch_size=batch_size)
  
  print('Train size: ' + str(len(train_gen) * batch_size))
  print('Validation size: ' + str(len(valid_gen) * batch_size))
  save_path = os.path.join(model_path,'weights.{epoch:02d}-{loss:.5f}.hdf5')
  # Checkpoint function for saving and logging progress
  checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
  # Function for logging progress to tensorboard
  tensorb = LRTensorBoard(log_dir=tensorboard_path)
  # Function for logging progress to CSV 
  logging = keras.callbacks.CSVLogger(logging_path)
  # Train model
  hist = model.fit_generator(train_gen, workers=6, use_multiprocessing=True, validation_data=valid_gen, shuffle=shuffle, epochs=nb_epochs,verbose=1, initial_epoch=init_epoch, callbacks=[checkpoint, tensorb, logging]) 

if predicting:
  # Load testing data
  ts = np.array(all_data['ts_test'])
  mp = np.zeros((len(ts) - matrix_profile_window + 1, 1))
  # Construct generator for test data
  test_gen =  MPTimeseriesGenerator(ts, mp, num_input_timeseries=n_input_series, internal_stride=channel_stride, num_outputs=num_outputs,lookahead=forward_sequences, lookbehind=lookbehind, important_upper_threshold=high_thresh, important_lower_threshold=low_thresh, important_weight=high_weight, length=input_width, mp_window=matrix_profile_window, stride=num_outputs, batch_size=128)
  predictions = model.predict_generator(test_gen, verbose=1, use_multiprocessing=True, workers=6)
  predictions = predictions.flatten()
  np.savetxt(R'predicted_matrix_profile.txt', predictions, fmt='%.10f')
  
