'''train dbow/dm for education/age/gender'''

import pandas as pd
from datetime import datetime
from collections import namedtuple
from gensim.models import Doc2Vec
import subprocess
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import codecs
import cfg
import numpy as np

#-------------------------prepare to train--------------------------------------------
def run_cmd(cmd):
    print(cmd)
    process = subprocess.Popen(cmd, shell=True,
                       stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
    for t, line in enumerate(iter(process.stdout.readline,b'')):
        line = line.decode('utf8').rstrip()
        print(line)
    process.communicate()
    return process.returncode

SentimentDocument = namedtuple('SentimentDocument', 'words tags')
class Doc_list(object):
    def __init__(self,f):
        self.f = f
    def __iter__(self):
        for i,line in enumerate(codecs.open(self.f,encoding='utf8')):
            words = line.split()
            tags = [int(words[0][2:])]
            words = words[1:]
            yield SentimentDocument(words,tags)



df_lb = pd.read_csv(cfg.data_path + 'all_v2.csv')
ys = {}
for lb in ['Education','age','gender']:
    ys[lb] = np.array(df_lb[lb])

#-------------------train dbow doc2vec---------------------------------------------
d2v = Doc2Vec(dm=0, size=200, negative=5, hs=0, min_count=3, window=30,sample=1e-5,workers=8,alpha=0.025,min_alpha=0.025)
doc_list = Doc_list('alldata-id.txt')
d2v.build_vocab(doc_list)


for i in range(2):
    print(datetime.now(),'pass:',i + 1)
    run_cmd('shuf alldata-id.txt > alldata-id-shuf.txt')
    doc_list = Doc_list('alldata-id.txt')
    d2v.train(doc_list)
    X_d2v = np.array([d2v.docvecs[i] for i in range(200000)])
    for lb in ["Education",'age','gender']:
        scores = cross_val_score(LogisticRegression(C=3),X_d2v,ys[lb],cv=5)
        print('dbow',lb,scores,np.mean(scores))
d2v.save(cfg.data_path + 'dbow_d2v.model')
print(datetime.now(),'save done')


#---------------train dm doc2vec-----------------------------------------------------
d2v = Doc2Vec(dm=1, size=200, negative=5, hs=0, min_count=3, window=10,sample=1e-5,workers=8,alpha=0.05,min_alpha=0.025)
doc_list = Doc_list('alldata-id.txt')
d2v.build_vocab(doc_list)

for i in range(10):
    print(datetime.now(),'pass:',i)
    run_cmd('shuf alldata-id.txt > alldata-id-shuf.txt')
    doc_list = Doc_list('alldata-id.txt')
    d2v.train(doc_list)
    X_d2v = np.array([d2v.docvecs[i] for i in range(200000)])
    for lb in ["Education",'age','gender']:
        scores = cross_val_score(LogisticRegression(C=3),X_d2v,ys[lb],cv=5)
        print('dm',lb,scores,np.mean(scores))
d2v.save(cfg.data_path + 'dm_d2v.model')
print(datetime.now(),'save done')