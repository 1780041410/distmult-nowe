# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, Value, Lock, Manager

import sys
if './src' not in sys.path:
    sys.path.append('./src')
    
import numpy as np
import os
import tensorflow as tf

import data  # we don't import individual things in a model. This is to make auto reloading in Notebook happy
from tester_transe import Tester
import logging
logging.basicConfig(filename='test_ConceptNet.log',format='%(message)s',level=logging.DEBUG)

path_prefix = './model/022618-FB15K/' #mkdir
model_path = path_prefix+'cn-distmult.ckpt'
data_path = path_prefix+'cn-data.bin'
filename = ["./data/FB15K/train.txt", "./data/FB15K/valid.txt"]
more_filt = "./data/FB15K/test.txt"

def test_link(tester, index, h_hit_flag, t_hit_flag,teston="ht", metric="hit", hitk=10, thread_num=8):
	test_num = len(tester.test_triples)
	if metric == "hit":
		# apply multithreading
		while index.value < len(tester.test_triples):
			triple = index.value
			index.value += 1
			print("PID: %d , Test id: %d Start" % (os.getpid(), triple))
			h,r,t = tester.test_triples[triple]
			h_topk = tester.link_pred_topk(t,r,target='h',topk=hitk)
			t_topk = tester.link_pred_topk(h,r,target='t',topk=hitk)
			print(h,[x.index for x in h_topk])
			print(t,[x.index for x in t_topk])
			if h in [x.index for x in h_topk]:
				logging.info('Triple ID:'+str(triple)+' Head:'+str(h))
				logging.info([x.index for x in h_topk])
				logging.info([x.dist for x in h_topk])
				h_hit_flag.append(1)
			if t in [x.index for x in t_topk]:
				logging.info('Triple ID:'+str(triple)+' Head:'+str(t))
				logging.info([x.index for x in t_topk])
				logging.info([x.dist for x in t_topk])
				t_hit_flag.append(1)
			if triple % 100 == 0:
			    print("test triples progress: {0} out of {1}".format(triple,test_num))
	else:
		raise ValueError("Invalid metric option!")

def printscore(h_hit_flag, t_hit_flag, test_num, teston="ht", hitk=10):
	if teston == 'h': # only h
		print("Hit@{0} on h-prediction: {1}".format(hitk, 1.0*sum(h_hit_flag)/test_num))
		logging.info("Hit@{0} on h-prediction: {1}".format(hitk, 1.0*sum(h_hit_flag)/test_num))
	elif teston == 't': # only t
		print("Hit@{0} on t-prediction: {1}".format(hitk, 1.0*sum(t_hit_flag)/test_num))
		logging.info("Hit@{0} on t-prediction: {1}".format(hitk, 1.0*sum(t_hit_flag)/test_num))
	else: # both h and t
		print("Hit@{0} on ht-prediction: {1}".format(hitk, 0.5*(sum(h_hit_flag)+sum(t_hit_flag))/test_num))
		logging.info("Hit@{0} on ht-prediction: {1}".format(hitk, 0.5*(sum(h_hit_flag)+sum(t_hit_flag))/test_num))

print("===== Step 4: Model testing ======")
m_test = Tester()
m_test.build(model_path,data_path)
m_test.load_test_data(more_filt)

# MultiProcess Manager(could store more info)
h_hit_flag = Manager().list() 
t_hit_flag = Manager().list()
index = Value('i', 0, lock=True)

thread_num = int(multiprocessing.cpu_count()/2)
logging.info("Using CPU core count:"+str(thread_num))
processes = [Process(target=test_link, args=(m_test, index, h_hit_flag, t_hit_flag)) for x in range(thread_num)]
for p in processes:
	p.start()
for p in processes:
	p.join()
printscore(h_hit_flag, t_hit_flag, len(m_test.test_triples), teston="ht", hitk=10)
# add parameter parser to _main_()
