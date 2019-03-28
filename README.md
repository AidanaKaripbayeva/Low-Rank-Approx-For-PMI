
# Low-Rank Approximation of Matrices for PMI-based Word Embeddings
A Python implementation of the paper:
"Low-Rank Approximation of Matrices for PMI-based Word Embeddings”
Alena Sorokina, Aidana Karipbayeva, Zhenisbek Assylbekov.
International Conference on Computational Linguistics and Intelligent Text Processing, 2019. 
Piblished in Lecture Notes in Computer Science (LNCS), Springer 2019. 

# Contacts
Authors: Aidana Karipbayeva, Alena Sorokina
Pull requests and issues: aidana.karipbayeva@nu.edu.kz; alena.sorokina@nu.edu.kz 

# Contents
We perform an empirical evaluation of several methods of low-rank approximation in the problem of obtaining PMI-based word embeddings. All word vectors were trained on parts of a large corpus (1B tokens) which was divided into equal-sized datasets, from which PMI matrices were obtained. A completely randomized design was used in assigning a method of low-rank approximation (SVD, NMF, QR) and a dimensionality of the vectors (250, 500) to each of the PMI matrix replicates. Our experiments show that word vectors obtained from the truncated SVD achieve the best performance on two downstream tasks, similarity and analogy, compare to the other two low-rank approximation methods.

Keywords: natural language processing, pointwise mutual information, matrix factorization, low-rank approximation, word vectors


The first file “Shifted Positive PMI” is used to obtain the PMI (Pointwise Mutual Information) Matrix. As input we used Enwik9 dataset  (http://mattmahoney.net/dc/Enwik9.zip), and as output we get two files: 
	SPPMI matrix in R^(V×V), where V is a vocabulary size. (your_pmi_name.npz)
	Vector in R^(1×V) , which contain the words as strings (your_column_name)

Then, we factorize SSPMI matrix, using different low-rank approximation methods, namely SVD, QR, NMF. We suggest to use as word embeddings:
	for SVD - U_d Σ_d^(1/2)
	for NMF - w_d
	for QR – (RU^T )_d


Obtained word embeddings (for each method) should be merged with the your_column_name file in the terminal. The instructions about this procedure are included in the each file for matrix factorization (SVD, QR, NMF). 

We evaluate the performance of the word embeddings using linguistic tasks: analogy and similarity. The Gensim library was used, and the code for evaluation is included in each file. 
