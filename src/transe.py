'''Model for holding TF parts. etc.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import data as pymod_data
from data import Data
import pickle

# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)

def orthogonal_initializer(scale=1.0, dtype=tf.float32):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)
  return _initializer

class TFParts(object):
    '''TensorFlow-related things.
    This is to keep TensorFlow-related components in a neat shell.
    '''

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, L1=False):
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        # margins
        self._m1 = 0.25
        self.L1 = L1
        self._neg_weight = 0.5
        self._epsilon = 1e-10
        self.build()

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def build(self):
        tf.reset_default_graph()
        

        with tf.variable_scope("graph", initializer=orthogonal_initializer()):
            # Variables (matrix of embeddings/transformations)

            self._ht = ht = tf.get_variable(
                name='ht',  # for t AND h
                shape=[self.num_cons, self.dim],
                dtype=tf.float32)
            self._r = r = tf.get_variable(
                name='r',
                shape=[self.num_rels, self.dim],
                dtype=tf.float32)

            
#             self._word_embedding = word_embedding = tf.get_variable(
#                 name='word_embedding',  # for word_embedding
#                 shape=[self.vocabulary_size, self.dim],
#                 dtype=tf.float32)
#             self._word_embedding_assign = word_embedding_assign = tf.placeholder(
#                 name='word_embedding_assign',
#                 shape=[self.vocabulary_size, self.dim],
#                 dtype=tf.float32)
            
            
            self._ht_assign = ht_assign = tf.placeholder(
                name='ht_assign',
                shape=[self.num_cons, self.dim],
                dtype=tf.float32)
            self._r_assign = r_assign = tf.placeholder(
                name='r_assign',
                shape=[self.num_rels, self.dim],
                dtype=tf.float32)

            # Type A loss : [|| h + r - t ||_2 + m1 - || h + r - t ||_2]+    here [.]+ means max (. , 0)
            self._A_h_index = A_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_h_index')
            self._A_r_index = A_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_r_index')
            self._A_t_index = A_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_t_index')
            
            self._A_neg_hn_index = A_neg_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(self.batch_size,self._neg_per_positive),
                name='A_neg_hn_index')
            self._A_neg_rel_hn_index = A_neg_rel_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(self.batch_size,self._neg_per_positive),
                name='A_neg_rel_hn_index')
            self._A_neg_t_index = A_neg_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=(self.batch_size,self._neg_per_positive),
                name='A_neg_t_index')
            self._A_neg_h_index = A_neg_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=(self.batch_size,self._neg_per_positive),
                name='A_neg_h_index')
            self._A_neg_rel_tn_index = A_neg_rel_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(self.batch_size,self._neg_per_positive),
                name='A_neg_rel_tn_index')
            self._A_neg_tn_index = A_neg_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(self.batch_size,self._neg_per_positive),
                name='A_neg_tn_index')
            
#             self._A_h_word_embedding_index = h_word_embedding_index = tf.placeholder(
#                 dtype = tf.int64,
#                 shape = [self.batch_size],
#                 name = 'h_word_embedding_index')
#             self._A_t_word_embedding_index = t_word_embedding_index = tf.placeholder(
#                 dtype = tf.int64,
#                 shape = [self.batch_size],
#                 name = 't_word_embedding_index')
            
            self._h_norm_batch = A_h_con_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht, A_h_index), 1)
            self._t_norm_batch = A_t_con_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht, A_t_index), 1)
            self._r_batch = A_rel_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(r, A_r_index), 1)
            
#             A_h_word_embedding_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(word_embedding, h_word_embedding_index), 1)
#             A_t_word_embedding_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(word_embedding, t_word_embedding_index), 1)
            
            A_neg_hn_con_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht,A_neg_hn_index), 2)
            A_neg_rel_hn_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(r,A_neg_rel_hn_index), 2)
            A_neg_t_con_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht,A_neg_t_index), 2)
            A_neg_h_con_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht,A_neg_h_index), 2)
            A_neg_rel_tn_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(r,A_neg_rel_tn_index), 2)
            A_neg_tn_con_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht,A_neg_tn_index), 2)
            
            print("A_neg_hn_con_batch:", A_neg_hn_con_batch.shape)

            
            f_score_h = tf.log(tf.sigmoid(tf.reduce_sum(tf.multiply(A_rel_batch, tf.multiply(A_h_con_batch, A_t_con_batch, "element_wise_multiply"),"r_product"),1)))
            f_score_hn = tf.scalar_mul(self._neg_weight,tf.reduce_mean(tf.log(tf.sigmoid((tf.add(tf.reduce_sum(tf.multiply(A_neg_rel_hn_batch, tf.multiply(A_neg_hn_con_batch, A_neg_t_con_batch)), 2), self._epsilon)))),1))
            f_score_tn = tf.scalar_mul(self._neg_weight, tf.reduce_mean(tf.log(tf.sigmoid((tf.add(tf.reduce_sum(tf.multiply(A_neg_rel_tn_batch, tf.multiply(A_neg_h_con_batch, A_neg_tn_con_batch)), 2), self._epsilon)))),1))
            
            self._A_loss = A_loss = (tf.reduce_sum(tf.subtract(tf.subtract(f_score_h, f_score_hn), f_score_tn)) ) / self._batch_size
            """
            f_score_h = tf.subtract(1., tf.sigmoid(tf.reduce_sum(tf.multiply(A_rel_batch, tf.multiply(A_h_con_batch, A_t_con_batch, "element_wise_multiply"),"r_product"),1)))
            f_score_hn = tf.sigmoid(tf.scalar_mul(self._neg_weight, tf.reduce_mean((tf.add(tf.reduce_sum(-tf.multiply(A_neg_rel_hn_batch, tf.multiply(A_neg_hn_con_batch, A_neg_t_con_batch)), 2), self._epsilon)),1)))
            f_score_tn = tf.sigmoid(tf.scalar_mul(self._neg_weight, tf.reduce_mean((tf.add(tf.reduce_sum(-tf.multiply(A_neg_rel_tn_batch, tf.multiply(A_neg_h_con_batch, A_neg_tn_con_batch)), 2), self._epsilon)),1)))
            
            self._A_loss = A_loss = (tf.reduce_sum(tf.add(tf.add(f_score_h, f_score_hn), f_score_tn)) ) / self._batch_size
            """
            # Optimizer
            self._lr = lr = tf.placeholder(tf.float32)
            # consider tf.train.AdagradOptimizer(lr)
            #self._opt = opt = tf.train.GradientDescentOptimizer(lr)
            self._opt = opt = tf.train.AdagradOptimizer(lr)
            self._train_op_A = train_op_A = opt.minimize(-A_loss)
            #remove
            #self._train_op_C_A = train_op_C_A = opt.minimize(C_loss_A)

            self._assign_ht_op = assign_ht_op = ht.assign(ht_assign)
            self._assign_r_op = assign_r_op = self._r.assign(r_assign)
            # self._assign_word_embedding_op = assign_word_embedding_op = word_embedding.assign(word_embedding_assign)
            # Saver
            self._saver = tf.train.Saver()
