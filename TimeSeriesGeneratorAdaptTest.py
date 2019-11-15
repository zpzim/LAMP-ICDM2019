import numpy as np
import pandas as pd
import os
import time
from TimeSeriesGeneratorAdapt import MPTimeseriesGenerator
from scipy.stats import zscore
ts = np.array(pd.read_csv(R'../data/merged/parkfield-20M-ts-train', sep=','))
mp = np.array(pd.read_csv(R'../data/merged/parkfield-20M-mp-train', sep=','))

lookahead=100
lookbehind=100
sample_length=100
length=100
outputs=100
stride=50
batch_size=128
internal_stride=25
n_input_dims=2

train_gen =  MPTimeseriesGenerator(ts, mp, length, num_input_timeseries=n_input_dims, lookahead=lookahead, lookbehind=lookbehind, num_outputs=outputs, mp_window=sample_length, stride=stride, batch_size=batch_size, internal_stride=internal_stride, recurrent=False)
index_list = [0,1,2,3,4,5,6,7,8,9,10, len(train_gen) - 1]





print('Testing correctness')
for batch_no in index_list:
  print(batch_no)
  data = train_gen[batch_no]
  x = data[0]
  y = data[1]
  print(x.shape)
  for batch_pos in range(min(batch_size, x.shape[0])):
    for i in range(n_input_dims):
      for j, k in enumerate(y[batch_pos,:,i]):
        if mp[stride*batch_size*batch_no + batch_pos*stride + lookbehind*internal_stride + j, i] != k:
          print(i, j, mp[stride*batch_size*batch_no + batch_pos*stride + lookbehind*internal_stride + j, i], k)
        assert(mp[stride*batch_size*batch_no + batch_pos*stride + lookbehind*internal_stride + j, i] == k)
    for i in range(n_input_dims):
      for j in range(lookahead+lookbehind):
        curr = stride*batch_size*batch_no + batch_pos*stride + j*internal_stride
        test = zscore(ts[curr:curr+length, i])
        if not np.all(x[batch_pos,j,:,i] == test):
          print('j = ', j, ' batch_pos = ', batch_pos)
          print(test, x[batch_pos,j,:,i])
        assert(np.all(x[batch_pos,j,:,i] == test))
print('Pass!')
print('Testing Performance')
start = time.time()
x = 0
for i in train_gen:
  x = x + 1
end = time.time()
print(end - start, ' seconds')
