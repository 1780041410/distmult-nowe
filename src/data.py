"""Processing of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import time
import tensorflow as tf

class Data(object):
    '''The abustrct class that defines interfaces for holding all data.
    '''

    def __init__(self):
        # concept vocab
        self.cons = []
        # rel vocab
        self.rels = []
        # transitive rels vocab
        self.index_cons = {}
        self.index_rels = {}
        # save triples as array of indices
        self.triples = np.array([0])
        self.triples_record = set([])
        self.neg_triples = np.array([0])
        # map for sigma
        # head per tail and tail per head (for each relation). used for bernoulli negative sampling
        self.hpt = np.array([0])
        self.tph = np.array([0])
        # recorded for tf_parts
        self.dim = 64
        self.batch_size = 1024
        self.L1=False

    def load_data(self, filename, splitter = '\t', line_end = '\n'):
        '''Load the dataset.'''
        triples = []
        last_c = -1
        last_r = -1
        hr_map = {}
        tr_map = {}
        
        for line in open(filename):
            line = line.rstrip(line_end).split('\t')
            if self.index_cons.get(line[0]) == None:
                self.cons.append(line[0])
                last_c += 1
                self.index_cons[line[0]] = last_c
            if self.index_cons.get(line[2]) == None:
                self.cons.append(line[2])
                last_c += 1
                self.index_cons[line[2]] = last_c
            if self.index_rels.get(line[1]) == None:
                self.rels.append(line[1])
                last_r += 1
                self.index_rels[line[1]] = last_r
            h = self.index_cons[line[0]]
            r = self.index_rels[line[1]]
            t = self.index_cons[line[2]]
            
            triples.append([h, r, t])
            self.triples_record.add((h, r, t))
        self.triples = np.array(triples)
        # calculate tph and hpt
        tph_array = np.zeros((len(self.rels), len(self.cons)))
        hpt_array = np.zeros((len(self.rels), len(self.cons)))
        for h,r,t in self.triples:
            tph_array[r][h] += 1.
            hpt_array[r][t] += 1.
        self.tph = np.mean(tph_array, axis = 1)
        self.hpt = np.mean(hpt_array, axis = 1)
        print("-- total number of entities:", len(self.cons))

    # add more triples to self.triples_record to 'filt' negative sampling
    def record_more_data(self, filename, splitter = '\t', line_end = '\n'):
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) < 3:
                continue
            h = self.con_str2index(line[0])
            r = self.rel_str2index(line[1])
            t = self.con_str2index(line[2])
            if h != None and r != None and t != None:
                self.triples_record.add((h, r, t))
        print("Loaded %s to triples_record." % (filename))
        print("Update: total number of triples in set:", len(self.triples_record))

    def num_cons(self):
        '''Returns number of ontologies.

        This means all ontologies have index that 0 <= index < num_onto().
        '''
        return len(self.cons)

    def num_rels(self):
        '''Returns number of relations.

        This means all relations have index that 0 <= index < num_rels().
        Note that we consider *ALL* relations, e.g. $R_O$, $R_h$ and $R_{tr}$.
        '''
        return len(self.rels)

    def rel_str2index(self, rel_str):
        '''For relation `rel_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_rels.get(rel_str)

    def rel_index2str(self, rel_index):
        '''For relation `rel_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.rels[rel_index]

    def con_str2index(self, con_str):
        '''For ontology `con_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_cons.get(con_str)

    def con_index2str(self, con_index):
        '''For ontology `con_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.cons[con_index]

    def rel(self):
        return np.array(range(self.num_rels()))

    def corrupt_pos(self, t, pos):
        hit = True
        res = None
        while hit:
            res = np.copy(t)
            samp = np.random.randint(self.num_cons())
            while samp == t[pos]:
                samp = np.random.randint(self.num_cons())
            res[pos] = samp
            if tuple(res) not in self.triples_record:
                hit = False
        return res


    #bernoulli negative sampling
    def corrupt(self, t, neg_per_positive, tar = None):
        res = []
        # print("array.shape:", res.shape)
        if tar == 't':
            for i in range(neg_per_positive):
                res.append(self.corrupt_pos(t, 2))
        elif tar == 'h':
            for i in range(neg_per_positive):
                res.append(self.corrupt_pos(t, 0))
        # else:
        #     this_tph = self.tph[t[1]]
        #     this_hpt = self.hpt[t[1]]
        #     assert(this_tph > 0 and this_hpt > 0)
        #     np.random.seed(int(time.time()))
        #     for i in range(neg_per_positive):
        #         if np.random.uniform(high=this_tph + this_hpt, low=0.) < this_hpt:
        #             res.extend(self.corrupt_pos(t, 2))
        #         else:
        #             res.extend(self.corrupt_pos(t, 0))
        return np.array(res)

    #bernoulli negative sampling on a batch
    def corrupt_batch(self, t_batch, neg_per_positive, tar = None):
        res = np.array([self.corrupt(t, neg_per_positive, tar) for t in t_batch])
        return res


    def save(self, filename):
        f = open(filename,'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)
