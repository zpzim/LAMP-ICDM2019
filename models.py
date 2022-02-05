from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Flatten, Reshape, Input, Dense, Activation, Dropout, MaxPooling2D, Permute, GlobalAveragePooling1D, concatenate, UpSampling1D, Conv2D, TimeDistributed, GlobalAveragePooling2D
import tensorflow as tf
import pandas as pd
import os,sys
import numpy as np

## Simple Resnet:
## Adapted from Time Series Classification with deep neural networks: A strong baseline
## Z. Wang, W. Yan, and T. Oates
def build_resnet_base(x, input_shape, n_feature_maps):
    fmt='channels_last'
    n_input_series = 1
    print ('build conv_x')
    conv_x = BatchNormalization()(x)
    conv_x = Conv2D(n_feature_maps, (8,1),  padding='same', data_format=fmt)(conv_x)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = Conv2D(n_feature_maps, (5,1), padding='same', data_format=fmt)(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = Conv2D(n_feature_maps, (3,1), padding='same', data_format=fmt)(conv_y)
    conv_z = BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = Conv2D(n_feature_maps, (1,1), padding='same', data_format=fmt)(x)
        shortcut_y = BatchNormalization()(shortcut_y)
    else:
        shortcut_y = BatchNormalization()(x)
    print ('Merging skip connection')
    y = tf.keras.layers.Add()([shortcut_y, conv_z])
    y = Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = Conv2D(n_feature_maps*2, (8,1), padding='same', data_format = fmt)(x1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = Conv2D(n_feature_maps*2, (5,1), padding='same', data_format=fmt)(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = Conv2D(n_feature_maps*2, (3,1), padding='same', data_format=fmt)(conv_y)
    conv_z = BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = Conv2D(n_feature_maps*2, (1,1), padding='same', data_format=fmt)(x1)
        shortcut_y = BatchNormalization()(shortcut_y)
    else:
        shortcut_y = BatchNormalization()(x1)
    print ('Merging skip connection')
    y = tf.keras.layers.Add()([shortcut_y, conv_z])
    y = Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = Conv2D(n_feature_maps*2, (8,1), padding='same', data_format=fmt)(x1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = Conv2D(n_feature_maps*2, (5,1), padding='same', data_format=fmt)(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = Conv2D(n_feature_maps*2, (3,1), padding='same', data_format=fmt)(conv_y)
    conv_z = BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = Conv2D(n_feature_maps*2, (1, 1), padding='same', data_format=fmt)(x1)
        shortcut_y = BatchNormalization()(shortcut_y)
    else:
        shortcut_y = BatchNormalization()(x1)
    print ('Merging skip connection')
    y = tf.keras.layers.Add()([shortcut_y, conv_z])
    y = Activation('relu')(y)
     
    print(y.shape)
    full = tf.keras.layers.GlobalAveragePooling2D(name='GlobalAvgPoolFinal', data_format=fmt)(y)   
    return full

def build_resnet(input_shape, n_feature_maps,  num_outputs):
  ip = Input(shape=input_shape)
  x = build_resnet_base(ip, input_shape, n_feature_maps)
  out = Dense(num_outputs, activation='sigmoid')(x)
  out = Reshape((num_outputs,1))(out)
  return Model(ip, out)

