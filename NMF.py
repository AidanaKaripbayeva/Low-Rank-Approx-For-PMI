
from sklearn.decomposition import NMF
import pandas as pd
import gensim
from gensim.models import KeyedVectors

import os
from scipy import sparse


dim = 500

sparse_pmim = sparse.load_npz("pmi_k5_10000.npz")

model = NMF(n_components=dim, init='nndsvda', random_state=0)
W = model.fit_transform(sparse_pmim)
H = model.components_


my_data = pd.DataFrame(W)
my_data.to_csv('your_nmf_vectors', sep=" ", index=False, header=False)

#Here, you should type to terminal:

#paste -d ' ' your_column_name your_nmf_vectors > your_nmf_file_for_gensim

#dim1, dim2 = shape of W matrix
#sed -i '1i dim1 dim2' your_nmf_file_for_gensim


model = gensim.models.KeyedVectors.load_word2vec_format('your_nmf_file_for_gensim', binary=False)
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
print(test_data_dir)
model.evaluate_word_pairs(test_data_dir +'wordsim353.tsv')

acc = model2.accuracy(test_data_dir + 'questions-words.txt')
corr_num = sum([len(section["correct"]) for section in acc])
incorr_num = sum([len(section["incorrect"]) for section in acc])
print(corr_num/(corr_num+incorr_num))
