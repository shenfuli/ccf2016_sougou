'''dm-nn stack for education/age/gender'''

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from gensim.models import Doc2Vec
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

from keras.utils import np_utils
import cfg


def accuracy(y_true, y_pred):
    '''

    计算正确率
    :param y_true:
    :param y_pred:
    :return:
    '''
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)


# -----------------------load dataset----------------------
df_all = pd.read_csv(cfg.data_path + 'all_v2.csv', encoding='utf8')
data_rows = df_all.shape[0]
print("data_rows = {}".format(data_rows))

ys = {}
for label in ['Education', 'age', 'gender']:
    ys[label] = np.array(df_all[label])

model = Doc2Vec.load(cfg.data_path + 'dm_d2v.model')
X_sp = np.array([model.docvecs[i] for i in range(data_rows)])

# ----------------------dmd2v stack for Education/age/gender---------------------------
df_stack = pd.DataFrame(index=range(data_rows))
TR = data_rows
n = 5

X = X_sp[:TR]
X_te = X_sp[TR:]

feat = 'dmd2v'
for i, lb in enumerate(['Education', 'age', 'gender']):
    num_class = len(pd.value_counts(ys[lb]))
    y = ys[lb][:TR]
    y_te = ys[lb][TR:]
    print("label={},train dataset shape row={},col={}".format(lb, X.shape[0], num_class))
    print("label={},test dataset shape row={},col={}".format(lb, X_te.shape[0], num_class))
    stack = np.zeros((X.shape[0], num_class))
    stack_te = np.zeros((X_te.shape[0], num_class))
    n = 5

    skf = StratifiedKFold(n_splits=n)
    for k, (tr, va) in enumerate(skf.split(X, y)):
        print('{} stack:{}/{}'.format(datetime.now(), k + 1, n))
        nb_classes = num_class
        X_train = X[tr]
        y_train = y[tr]
        X_va = X[va]
        y_va = y[va]

        X_train = X_train.astype('float32')
        X_va = X_va.astype('float32')
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_va = np_utils.to_categorical(y_va, nb_classes)

        model = Sequential()
        model.add(Dense(200, input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.1))
        model.add(Activation('tanh'))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        history = model.fit(X_train, Y_train, shuffle=True,
                            batch_size=128, nb_epoch=35,
                            verbose=2, validation_data=(X_va, Y_va))
        y_pred_va = model.predict_proba(X[va])
        y_pred_te = model.predict_proba(X_te)
        print('va acc:', accuracy(y[va], y_pred_va))
        print('te acc:', accuracy(y_te, y_pred_te))
        stack[va] += y_pred_va
        stack_te += y_pred_te
    stack_te /= n
    stack_all = np.vstack([stack, stack_te])
    for l in range(stack_all.shape[1]):
        df_stack['{}_{}_{}'.format(feat, lb, l)] = stack_all[:, l]
df_stack.to_csv(cfg.data_path + 'dmd2v_stack_10W.csv', encoding='utf8', index=None)
print(datetime.now(), 'save dmd2v stack done!')
