'''tfidf-lr stack for education/age/gender'''
import pandas as pd
import numpy as np
import jieba
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import cfg
import warnings
warnings.filterwarnings('ignore')

def accuracy(y_true, y_pred):
    '''

    计算正确率
    :param y_true:
    :param y_pred:
    :return:
    '''
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)

class Tokenizer():
    '''
        分词后，bi-gram特征提取
    '''
    
    def __init__(self):
        self.n = 0

    def __call__(self, line):
        tokens = []
        for query in line.split('\t'):
            words = [word for word in jieba.cut(query)]
            for gram in [1, 2]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i + gram])]
        if np.random.rand() < 0.00001:
            print(line)
            print('=' * 20)
            print(tokens)
        self.n += 1
        if self.n % 10000 == 0:
            print(self.n)
        return tokens


df_all = pd.read_csv(cfg.data_path + 'all_v2.csv', encoding='utf8')
ys = {}
for label in ['Education', 'age', 'gender']:
    ys[label] = np.array(df_all[label])

train_rows = df_all.shape[0]
print('train_rows={}'.format(train_rows))


tfv = TfidfVectorizer(tokenizer=Tokenizer(), min_df=3, max_df=0.95, sublinear_tf=True)
X_sp = tfv.fit_transform(df_all['query'])
pickle.dump(X_sp, open(cfg.data_path + 'tfidf_10W.feat', 'wb'))

df_stack = pd.DataFrame(index=range(len(df_all)))
# -----------------------stack for education/age/gender------------------
for lb in ['Education', 'age', 'gender']:
    num_class = np.unique(ys[lb])
    TR = int(train_rows * 0.8)
    print("label={},num_class={},train_size = {},test_size={}".format(lb,num_class,TR,int(train_rows - TR)))

    X, X_te = X_sp[:TR], X_sp[TR:]
    y, y_te = ys[lb][:TR], ys[lb][TR:]

    stack = np.zeros((X.shape[0], num_class))
    stack_te = np.zeros((X_te.shape[0], num_class))

    n = 5
    skf = StratifiedKFold(n_splits=n)
    for i, (tr, va) in enumerate(skf.split(X, y)):
        print('%s stack:%d/%d' % (str(datetime.now()), i + 1, n))
        clf = LogisticRegression(C=3)
        clf.fit(X[tr], y[tr])
        y_pred_va = clf.predict_proba(X[va])
        y_pred_te = clf.predict_proba(X_te)
        print('va acc:', accuracy(y[va], y_pred_va))
        print('te acc:', accuracy(y_te, y_pred_te))
        stack[va] += y_pred_va
        stack_te += y_pred_te

    stack_te /= n
    stack_all = np.vstack([stack, stack_te])
    for i in range(stack_all.shape[1]):
        df_stack['tfidf_{}_{}'.format(lb, i)] = stack_all[:, i]

df_stack.to_csv(cfg.data_path + 'tfidf_stack_10W.csv', index=None, encoding='utf8')
print(datetime.now(), 'save tfidf stack done!')
