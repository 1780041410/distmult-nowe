# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm

import sys
if '../src' not in sys.path:
    sys.path.append('../src')
    
import numpy as np
import os
import tensorflow as tf

import data  # we don't import individual things in a model. This is to make auto reloading in Notebook happy
import transe
import trainer_transe
from trainer_transe import Trainer

model_path = 'xcn-transe.ckpt'
data_path = 'xcn-data.bin'
filename = "../data/conceptnetCDTrain.tsv"
more_filt = ['../data/conceptnetCDTest.tsv']

# #####----word2vec part-----
# vocabulary_size=50000
# text_file = '/zf2/jz4fu/Github/word2vec/text8'
# w2v = word2vec_basic.Word2vec(vocabulary_size=50000, batch_size= 128, embedding_size= 128, skip_window=1, num_skips=2, num_sampled=64, num_steps=10001)
# vocabulary = w2v.read_data(text_file)
# text_data, count, word2id, id2word = w2v.build_dataset(text_file)
# del vocabulary  # Hint to reduce memory.
# print('Most common words (+UNK)', count[:5])
# print('Sample data', text_data[:10], [id2word[i] for i in text_data[:10]])
# batch, labels = w2v.generate_batch(text_data,8,2,1)
# final_embeddings = w2v.build_and_train(text_data, None, 1)
# np.append(final_embeddings, [[0.]*final_embeddings.shape[1]], axis = 0)
# print("word embedding shape:", final_embeddings.shape)
# # print(final_embeddings[word2id['collaborative']])

this_data = data.Data()
this_data.load_data(filename=filename)
for f in more_filt:
    this_data.record_more_data(f)


m_train = Trainer()
m_train.build(this_data,  dim=128, batch_size=500, save_path = model_path, data_save_path = data_path, L1=False)

ht_embedding = m_train.train(epochs=50, save_every_epoch=100, lr=0.01, a1=1., m1=.5)
