# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm

import sys
if './src' not in sys.path:
    sys.path.append('./src')
    
import numpy as np
import os
import tensorflow as tf

import data  # we don't import individual things in a model. This is to make auto reloading in Notebook happy
import transe
import trainer_transe
from trainer_transe import Trainer


path_prefix = './model/022618-FB15K/' #mkdir
if not os.path.exists(path_prefix):
    os.makedirs(path_prefix)
model_path = path_prefix+'cn-distmult.ckpt'
data_path = path_prefix+'cn-data.bin'
filename = ["./data/FB15K/test.txt", "./data/FB15K/valid.txt"]
more_filt = ["./data/FB15K/test.txt"]


this_data = data.Data()
for f in filename:
	this_data.load_data(f,splitter=' ')
for f in more_filt:
    this_data.record_more_data(f,splitter=' ')



m_train = Trainer()
m_train.build(this_data,  dim=50, batch_size=100, neg_per_positive=64, save_path = model_path, data_save_path = data_path, L1=False)


print("===== Step 3: Model building =====")
ht_embedding,r_embedding = m_train.train(epochs=250, save_every_epoch=10, lr=0.01, a1=1., m1=.5)
np.save(path_prefix+'ht_embedding.npy', ht_embedding)
np.save(path_prefix+'r_embedding.npy', r_embedding)

np.save(path_prefix+'trainloss_hist.npy', np.array(m_train.trainloss_hist))
np.save(path_prefix+'posneg_hist.npy', np.array(m_train.posneg_hist))
# add parameter parser to _main_()
