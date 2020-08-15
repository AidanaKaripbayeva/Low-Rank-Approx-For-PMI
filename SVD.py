from scipy import sparse
from scipy.sparse.linalg import svds
import gensim
from gensim.models import KeyedVectors

import tensorflow as tf
import numpy as np
import os
import pandas as pd

rng = np.random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)


dim = 500

sparse_pmim = sparse.load_npz("pmi_k5_10000.npz")

def svd_emb(dim):
    u, s, vt = svds(sparse_pmim, k=dim)
    S = np.diag(s)

    with tf.Session(config=config) as sess:
        s_sqrt = tf.sqrt(S)
        embed = tf.matmul(u,s_sqrt)

    sess= tf.Session()
    e = sess.run(embed)
    emb = pd.DataFrame(e)
    return emb

my_data = svd_emb(dim)
my_data.to_csv('your_svd_vectors', sep=" ", index=False, header=False)


# Here, you should type to terminal:
#
# paste -d ' ' your_column_name your_svd_vectors > your_svd_file_for_gensim
#
# dim1, dim2 = shape of e matrix
# sed -i '1i dim1 dim2' your_svd_file_for_gensim

model = gensim.models.KeyedVectors.load_word2vec_format('your_svd_file_for_gensim', binary=False)
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
print(test_data_dir)
model.evaluate_word_pairs(test_data_dir +'wordsim353.tsv')
acc = model2.accuracy(test_data_dir + 'questions-words.txt')
corr_num = sum([len(section["correct"]) for section in acc])
incorr_num = sum([len(section["incorrect"]) for section in acc])
print(corr_num/(corr_num+incorr_num))
