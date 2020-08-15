import pandas as pd
import gensim
from gensim.models import KeyedVectors
from scipy import random, linalg, dot, diag, all, allclose, sparse

dim = 500

sparse_pmim = sparse.load_npz("pmi_k5_10000.npz").toarray()

q, r, p = linalg.qr(sparse_pmim, mode='economic', pivoting= True)

R_sored = r[:, np.argsort(p)]
R_sorted_dim = np.transpose(R_sorted[:dim,:])

my_data = pd.DataFrame(R_sorted_dim)
my_data.to_csv('your_qr_vectors', sep=" ", index=False, header=False)

#Here, you should type to terminal:
#paste -d ' ' your_column_name your_qr_vectors > your_qr_file_for_gensim
#In the next line, dim1, dim2 = shape of R matrix
#sed -i '1i dim1 dim2' your_qr_file_for_gensim

model = gensim.models.KeyedVectors.load_word2vec_format('your_qr_file_for_gensim', binary=False)

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
print(test_data_dir)
model.evaluate_word_pairs(test_data_dir +'wordsim353.tsv')

acc = model2.accuracy(test_data_dir + 'questions-words.txt')
corr_num = sum([len(section["correct"]) for section in acc])
incorr_num = sum([len(section["incorrect"]) for section in acc])
print(corr_num/(corr_num+incorr_num))
