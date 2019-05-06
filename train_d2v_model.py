'''train dbow/dm for education/age/gender'''
# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime
from collections import namedtuple
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import codecs
import cfg
import numpy as np

SentimentDocument = namedtuple('SentimentDocument', 'words tags')


class Doc_list(object):
    def __init__(self, f):
        self.f = f

    def __iter__(self):
        for i, line in enumerate(codecs.open(self.f, encoding='utf8')):
            words = line.split()
            tags = [int(words[0][2:])]
            words = words[1:]
            # data format: SentimentDocument(words=['柔和', '双沟', '女生', '中财网', '首页', '财经'], tags=[0])
            yield SentimentDocument(words, tags)


df_lb = pd.read_csv(cfg.data_path + 'all_v2.csv')
data_rows = df_lb.shape[0]
print("data_rows={}".format(data_rows))

ys = {}
for lb in ['Education', 'age', 'gender']:
    ys[lb] = np.array(df_lb[lb])

# -------------------train dbow doc2vec---------------------------------------------
d2v = Doc2Vec(dm=0, vector_size=200, negative=5, hs=0, min_count=3, window=5, sample=1e-5, workers=8, alpha=0.025,
              min_alpha=0.025)
doc_list = Doc_list('alldata-id.txt')
d2v.build_vocab(doc_list)

d2v.train(doc_list, total_examples=data_rows, epochs=2)
X_d2v = np.array([d2v.docvecs[i] for i in range(data_rows)])
for lb in ["Education", 'age', 'gender']:
    scores = cross_val_score(LogisticRegression(C=3), X_d2v, ys[lb], cv=5)
    print('dbow', lb, scores, np.mean(scores))
d2v.save(cfg.data_path + 'dbow_d2v.model')
print(datetime.now(), 'save done')

# ---------------train dm doc2vec-----------------------------------------------------
d2v = Doc2Vec(dm=1, vector_size=200, negative=5, hs=0, min_count=3, window=10, sample=1e-5, workers=8, alpha=0.05,
              min_alpha=0.025)
doc_list = Doc_list('alldata-id.txt')
d2v.build_vocab(doc_list)

d2v.train(doc_list, total_examples=data_rows, epochs=2)
X_d2v = np.array([d2v.docvecs[i] for i in range(data_rows)])
for lb in ["Education", 'age', 'gender']:
    scores = cross_val_score(LogisticRegression(C=3), X_d2v, ys[lb], cv=5)
    print('dm', lb, scores, np.mean(scores))
d2v.save(cfg.data_path + 'dm_d2v.model')
print(datetime.now(), 'save done')


'''
dbow Education [0.51119888 0.51854815 0.52777361 0.53520352 0.53353003] 0.525250837294673
dbow age [0.51817227 0.5280736  0.5409     0.5516     0.55866173] 0.539481520565106
dbow gender [0.79166042 0.79065    0.80155    0.80805    0.80229011] 0.7988401062969753
2019-05-06 12:01:47.366473 save done
dm Education [0.49725027 0.51189881 0.51327434 0.51480148 0.50372556] 0.5081900920713034
dm age [0.50322452 0.5239738  0.5308     0.52745    0.51910382] 0.5209104276793278
dm gender [0.75221239 0.77615    0.7828     0.7746     0.76768838] 0.7706901547599504
'''

