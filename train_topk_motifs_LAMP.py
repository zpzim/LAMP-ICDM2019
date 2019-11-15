import numpy as np
import scipy.io as sio
import pandas as pd
from scipy.stats import pearsonr
from multiprocessing import Pool
from itertools import product
import sys,os
import numba


# MP window size
window = 100

# Percent of the subsequences to look at
topk_percent = 5

motif_thresh = 0.75

corr_thresh = 0.95

predict_num_threads = 6
predict_chunksize = 10000

@numba.jit
def corr(query, model, query_mean, query_std, model_means, model_stds):
    # Generate the dot products between the query and the model
    #dot_products = np.matmul(model, query)
    #dot_products = model @ query
    dot_products = np.dot(model, query)
    cross_mean = dot_products / query.size
    return (cross_mean - query_mean * model_means) / (query_std * model_stds)

def remaining_checks(i, to_keep):
  return np.count_nonzero(to_keep[i+1:]) != 0


def kdiversification(candidates, k):
  model = np.zeros((k, candidates.shape[1]))
  means = np.mean(candidates, axis=1)
  stds = np.std(candidates, axis=1)
  model[0,:] = candidates[0,:]
  correlation = corr(model[0, :], candidates, means[0], stds[0], means, stds)
  curr = 1
  while curr < k:
    index = np.argmin(correlation)
    model[curr, :] = candidates[index, :]
    newcorrs = corr(model[curr,:], candidates, means[index], stds[index], means, stds)
    correlation = np.maximum(correlation, newcorrs)
    curr += 1
    if curr % 100 == 0:
      print(curr)
  return model


@numba.jit
def refine_iter(model, to_keep, i, means, stds, thresh):
  nonzero = np.nonzero(to_keep[i+1:])[0] + i + 1
  c = corr(model[i,:], model[nonzero, :], means[i], stds[i], means[nonzero], stds[nonzero])
  to_keep[nonzero[c>thresh]] = False
  return to_keep

# Takes a model (each column is a motif extracted via the MP) and removes columns which are highly correlated
# Returns a refined model
def refine_model(model, means, stds, thresh):
  i = 0
  to_keep = np.ones((model.shape[0],), dtype=bool)
  while i < model.shape[0] and remaining_checks(i, to_keep):
    if to_keep[i]:
      to_keep = refine_iter(model, to_keep, i, means, stds, thresh)
    i += 1
    print(i, len(to_keep) - np.count_nonzero(to_keep))
  model = model[to_keep, :]
  means = means[to_keep]
  stds = stds[to_keep]
  return model, means, stds

data_path = sys.argv[1];

all_data = sio.loadmat(sys.argv[1])


model_path = None
if len(sys.argv) > 2:
  model_path = sys.argv[2]
  


def do_kdiversification(all_data, model_path):
  if model_path is None:
    mp = all_data['mp_train'].flatten()
    ts = all_data['ts_train'].flatten()
    #Sort MP in decending order
    idxs = np.argsort(-mp)
    last_index = np.argwhere(mp[idxs] > motif_thresh)[-1]
    idxs = idxs[:last_index[0]]
    #Populate model with subsequences
    candidates = np.zeros((len(idxs), window))
    for i, elem in enumerate(idxs):
      candidates[i, :] = ts[elem:elem+window]
    model = kdiversification(candidates, topk_motifs)
    #Precompute the model means and stds
    means = np.mean(model, axis=1)
    stds = np.std(model, axis=1)
    #Remove correlated motifs
    np.savetxt('baseline_model.txt', model)
  else:
    # Load the pre-refined model
    model = np.loadtxt(model_path)
    # Precompute the means/stds
    means = np.mean(model, axis=1)
    stds = np.std(model, axis=1)
    
  return model, means, stds


def do_simple_diversemotifs(all_data, model_path):

  if model_path is None:
    mp = all_data['mp_train'].flatten()
    ts = all_data['ts_train'].flatten()
    #Sort MP in decending order
    idxs = np.argsort(-mp)
    #Take the best motifs
    topk_amt = int(len(idxs) * (topk_percent / 100))
    idxs = idxs[:topk_amt]
    #Populate model with subsequences
    model = np.zeros((len(idxs), window))
    for i, elem in enumerate(idxs):
      model[i, :] = ts[elem:elem+window]

    #Precompute the model means and stds
    means = np.mean(model, axis=1)
    stds = np.std(model, axis=1)
    #Remove correlated motifs
    model, means, stds = refine_model(model, means, stds, corr_thresh)
    np.savetxt('baseline_model.txt', model)
  else:
    # Load the pre-refined model
    model = np.loadtxt(model_path)
    # Precompute the means/stds
    means = np.mean(model, axis=1)
    stds = np.std(model, axis=1)
  return model, means, stds

# Simple diverse motifs model
model, means, stds = do_simple_diversemotifs(all_data, model_path)

# K-diverseification test
#model, means, stds = do_kdiversification(all_data, model_path)


ts_test = all_data['ts_val'].flatten()


@numba.jit
def model_predict(i):
  sequence = ts_test[i:i+window]
  c = corr(sequence, model, np.mean(sequence), np.std(sequence), means, stds)
  return np.max(c)


predictions = np.zeros((len(ts_test) - window + 1))
pool = Pool(predict_num_threads)
for ind, res in enumerate(pool.imap(model_predict, range(len(predictions)), predict_chunksize)):
  predictions[ind] = res
  if ind % 10000 == 0:
    print(ind, predictions[ind])

np.savetxt('baseline_predictions.txt', predictions)
