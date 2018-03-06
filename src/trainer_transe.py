''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time

import data
import transe


class Trainer(object):
    def __init__(self):
        self.batch_size=128
        self.neg_per_positive = 10
        self.dim=64
        self.this_data = None
        self.tf_parts = None
        self.save_path = 'this-transe.ckpt'
        self.data_save_path = 'this-data.bin'
        self.L1=False

    def build(self, data, dim=64, neg_per_positive = 64, batch_size=128, save_path = 'this-transe.ckpt', data_save_path = 'this-data.bin', L1=False):
        self.this_data = data
        self.neg_per_positive = neg_per_positive
        self.dim = self.this_data.dim = dim
        self.batch_size = self.this_data.batch_size = batch_size
        self.data_save_path = data_save_path
        self.save_path = save_path
        self.L1 = self.this_data.L1 = L1
        self.tf_parts = transe.TFParts(num_rels=self.this_data.num_rels(),
                                 num_cons=self.this_data.num_cons(),
                                 dim=dim,
                                 batch_size=self.batch_size,
                                 neg_per_positive = self.neg_per_positive,
                                 L1=self.L1)

    def gen_A_batch(self, forever=False, shuffle=True):
        l = self.this_data.triples.shape[0]
        while True:
            triples = self.this_data.triples
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, self.batch_size):
                batch = triples[i: i+self.batch_size, :]
                if batch.shape[0] < self.batch_size:
                    batch = np.concatenate((batch, self.this_data.triples[:self.batch_size - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_size
                all_neg_hn_batch = self.this_data.corrupt_batch(batch, self.neg_per_positive, "h")
                all_neg_tn_batch = self.this_data.corrupt_batch(batch, self.neg_per_positive, "t")
                h_batch, r_batch, t_batch = batch[:, 0], batch[:, 1], batch[:, 2]

                neg_hn_batch, neg_rel_hn_batch, neg_t_batch, neg_h_batch, neg_rel_tn_batch, neg_tn_batch = all_neg_hn_batch[:, :,0], all_neg_hn_batch[:, :, 1], all_neg_hn_batch[:,:, 2], all_neg_tn_batch[:, :,0],all_neg_tn_batch[:, :, 1], all_neg_tn_batch[:,:, 2]
                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(np.int64), neg_hn_batch.astype(np.int64), neg_rel_hn_batch.astype(np.int64), neg_t_batch.astype(np.int64),neg_h_batch.astype(np.int64),neg_rel_tn_batch.astype(np.int64), neg_tn_batch.astype(np.int64)
            if not forever:
                break

    def train(self, epochs=20, save_every_epoch=10, lr=0.001, a1=0.1, m1=0.5, splitter='\t', line_end='\n'):
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        
#         sess.run([self.tf_parts._assign_word_embedding_op],
#                 feed_dict={
#                     self.tf_parts._word_embedding_assign : self.word_embedding
#                 })

        num_A_batch = len(list(self.gen_A_batch()))
        #num_C_batch = len(list(gen_C_batch(self.this_data, self.batch_size)))
        
        # margins
        self.tf_parts._m1 = m1
        t0 = time.time()
        for epoch in range(epochs):
            epoch_loss = self.train1epoch(sess, num_A_batch, lr, a1 , m1, epoch + 1)
            print("Time use: %d" % (time.time() - t0))
            if np.isnan(epoch_loss):
                print("Training collapsed.")
                return
            if (epoch + 1) % save_every_epoch == 0:
                this_save_path = self.tf_parts._saver.save(sess, self.save_path)
                self.this_data.save(self.data_save_path)
                print("transe saved in file: %s. Data saved in file: %s" % (this_save_path, self.data_save_path))
    
        this_save_path = self.tf_parts._saver.save(sess, self.save_path)
        with sess.as_default():
            ht_embeddings = self.tf_parts._ht.eval()
            r_embeddings = self.tf_parts._r.eval()
        print("transe saved in file: %s" % this_save_path)
        sess.close()
        print("Done")
        return ht_embeddings,r_embeddings

    def train1epoch(self, sess, num_A_batch, lr, a1, m1, epoch, debug=True):
        '''build and train a model.

        Args:
            self.batch_size: size of batch
            num_epoch: number of epoch. A epoch means a turn when all A/B_t/B_h/C are passed at least once.
            dim: dimension of embedding
            lr: learning rate
            self.this_data: a Data object holding data.
            save_every_epoch: save every this number of epochs.
            save_path: filepath to save the tensorflow model.
        '''

        this_gen_A_batch = self.gen_A_batch(forever=True)

        this_loss = []

        loss_A = loss_C_A = 0
        
        for batch_id in range(num_A_batch):
            # Optimize loss A
            A_h_index, A_r_index, A_t_index, A_neg_hn_index, A_neg_rel_hn_index, A_neg_t_index,A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index  = next(this_gen_A_batch)
            _, loss_A = sess.run([self.tf_parts._train_op_A, self.tf_parts._A_loss],
                    feed_dict={self.tf_parts._A_h_index: A_h_index,
                               self.tf_parts._A_r_index: A_r_index,
                               self.tf_parts._A_t_index: A_t_index,
                               self.tf_parts._A_neg_hn_index: A_neg_hn_index,
                               self.tf_parts._A_neg_rel_hn_index: A_neg_rel_hn_index,
                               self.tf_parts._A_neg_t_index: A_neg_t_index,
                               self.tf_parts._A_neg_h_index: A_neg_h_index,
                               self.tf_parts._A_neg_rel_tn_index: A_neg_rel_tn_index,
                               self.tf_parts._A_neg_tn_index: A_neg_tn_index,
                               # self.tf_parts._A_h_word_embedding_index: A_h_word_embedding_index,
                               # self.tf_parts._A_t_word_embedding_index: A_t_word_embedding_index,
                               self.tf_parts._lr: lr})
            """
            _, loss_C_A = sess.run([self.tf_parts._train_op_C_A, self.tf_parts._C_loss_A],
                    feed_dict={self.tf_parts._A_h_index: A_h_index,
                               self.tf_parts._A_r_index: A_r_index,
                               self.tf_parts._A_t_index: A_t_index,
                               self.tf_parts._A_neg_hn_index: A_neg_hn_index,
                               self.tf_parts._A_neg_rel_hn_index: A_neg_rel_hn_index,
                               self.tf_parts._A_neg_t_index: A_neg_t_index,
                               self.tf_parts._A_neg_h_index: A_neg_h_index,
                               self.tf_parts._A_neg_rel_tn_index: A_neg_rel_tn_index,
                               self.tf_parts._A_neg_tn_index: A_neg_tn_index,
                               self.tf_parts._lr: lr * a1})
            """

            # Observe total loss
            batch_loss = [loss_A]
            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)

            if ((batch_id + 1) % 50 == 0) or batch_id == num_A_batch - 1:
                print('process: %d / %d. Epoch %d' % (batch_id+1, num_A_batch, epoch))
        
        if debug:
            h_debug, t_debug, r_debug = sess.run([self.tf_parts._h_norm_batch, self.tf_parts._t_norm_batch, self.tf_parts._r_batch],
                    feed_dict={self.tf_parts._A_h_index: A_h_index,
                               self.tf_parts._A_r_index: A_r_index,
                               self.tf_parts._A_t_index: A_t_index})
            h_ndebug, t_ndebug, r_ndebug = sess.run([self.tf_parts._h_norm_batch, self.tf_parts._t_norm_batch, self.tf_parts._r_batch],
                    feed_dict={self.tf_parts._A_h_index: A_neg_hn_index[:,0],
                               self.tf_parts._A_r_index: A_neg_rel_hn_index[:,0],
                               self.tf_parts._A_t_index: A_neg_t_index[:,0]})
            debug_pos_loss = np.sum(np.multiply(r_debug, np.multiply(h_debug, t_debug))) / len(h_debug)
            debug_neg_loss = np.sum(np.multiply(r_ndebug, np.multiply(h_ndebug, t_ndebug))) / len(h_debug)
            print('pos_loss=',debug_pos_loss,'\nneg_loss=',debug_neg_loss)

        this_total_loss = np.sum(this_loss)
        print("Loss of epoch %d = %s" % (epoch, np.sum(this_total_loss)))
        print([l for l in this_loss])
        return this_total_loss

# A safer loading is available in Tester, with parameters like batch_size and dim recorded in the corresponding Data component
def load_tfparts(batch_size = 128,
                dim = 64,
                this_data=None,
                save_path = 'this-transe.ckpt', L1=False):
    tf_parts = transe.TFParts(num_rels=this_data.num_rels(),
                             num_cons=this_data.num_cons(),
                             dim=dim,
                             batch_size=self.batch_size, L1=L1)
    with tf.Session() as sess:
        tf_parts._saver.restore(sess, save_path)
