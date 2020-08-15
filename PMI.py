from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile
import gensim
import scipy.sparse
import string

import numpy as np
import pandas as pd
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from scipy.sparse import coo_matrix
from scipy import sparse

from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.utils.extmath import randomized_svd

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import svds, eigs
from numpy.linalg import solve, norm
from numpy.random import rand
from gensim.models import Word2Vec
from collections import Counter
from math import log


rng = np.random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)

#hyperparameters
vocab_size = 10000
window_size = 2
k = 5 #size of the negative sample

def maybe_download(filename, url, path):
    """Download a file if not present, and make sure it's the right size."""
    local_filename = os.path.join(path, filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename)
    statinfo = os.stat(local_filename)
    return local_filename

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0]))
    return data

# Clean the data
def clean_data():
    filename = maybe_download('text8.zip', 'http://mattmahoney.net/dc/', '/home/alena/')
    vocabulary = read_data(filename)
    words = vocabulary.split()

    # convert to lower case
    words = [word.lower() for word in words]

    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    return words

def build_dataset(words, vocab_sizze):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_sizze-1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def gen_bigrams(data, window_sizze):
    for idx in range(len(data)):

        if idx==1:
            window = data[idx-1:idx+window_sizze]
            w = window[1]
            for i in range(len(window)):
                if i == 1:
                    continue
                yield(w, window[i])


        elif idx == 0:
            window = data[idx: idx + window_sizze]
            w = window[0]
            for i in range(len(window)):
                if i == 0:
                    continue
                yield(w, window[i])


        else:
            window = data[idx-2: idx+window_sizze]
            w = window[2]
            for i in range(len(window)):
                if i == 2:
                    continue
                yield(w, window[i])



def construct_vocab(data):
    vocab = Counter()

    for (w1, w2) in gen_bigrams(data, window_size): # count 1gram & 2gram
        vocab.update([w1, w2, (w1, w2)])
    return vocab


def calc_pmi(vocab, det):

    for (w1, w2) in filter(lambda el: isinstance(el, tuple), vocab):
        p_a, p_b = float(vocab[w1]), float(vocab[w2])
        p_ab = float(vocab[(w1, w2)])
        pmi = log((det * p_ab) / (p_a * p_b), 2)
        sppmi = max(pmi - log(k,10), 0)
        sppmi=sppmi/2
        yield (w1, w2, sppmi)

def calc_det(vocabb):
    det = 0.0
    for (w1,w2) in  filter(lambda el: isinstance(el, tuple), vocabb):
        det = det + float(vocabb[(w1,w2)])
    return det

def constr_sparse_pmi():

    row=[]
    column=[]
    data = []

    for (w1,w2, sppmi) in calc_pmi(vocab, calc_det(vocab)):
        row.append(w1)
        column.append(w2)
        data.append(sppmi)
        row.append(w2)
        column.append(w1)
        data.append(sppmi)
    sparse_pmi = sparse.csr_matrix((data, (row, column)), shape=(len(dictionary), len(dictionary)))

    return sparse_pmi

def col():
    column = []
    for i in reversed_dictionary:
        column.append(reversed_dictionary[i])
    my_col = pd.DataFrame(column)
    return my_col


data, count, dictionary, reversed_dictionary = build_dataset(clean_data(), vocab_size)
vocab = construct_vocab(data)
sparse_pmim = constr_sparse_pmi()

sparse.save_npz("pmi_k"+str(k)+"_"+str(vocab_size)+".npz", sparse_pmim)
my_data = col()
my_data.to_csv("c_k"+str(k)+"_"+str(vocab_size), sep=" ", index=False, header=False)
