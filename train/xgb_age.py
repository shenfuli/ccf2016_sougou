'''xgb-ens for education/age/gender'''
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
import cfg
lb = 'gender'
model_name = 'xgb_gender.model'

def xgb_acc_score(preds, dtrain):
    y_true = dtrain.get_label()
    y_pred = np.argmax(preds, axis=1)
    return [('acc', np.mean(y_true == y_pred))]

df_lr = pd.read_csv('./data/tfidf_stack_10W.csv')
df_dbow = pd.read_csv('./data/dbowd2v_stack_10W.csv')
df_lr_cols = [col for col in df_lr.columns if lb not in col ]
df_dbow_cols = [col for col in df_dbow.columns if lb not in col ]
df_lr = df_lr[df_lr_cols]
df_dbow = df_dbow[df_dbow_cols]

data_df = pd.read_csv(cfg.data_path + 'all_v2.csv')

y = np.array(data_df[lb])
print(data_df.head())
print(y)

df = pd.concat([df_lr, df_dbow], axis=1)
print(df.head())

TR = 80000
esr = 100
evals = 1
num_class = len(np.unique(y))

X ,X_te= df.iloc[:TR], df.iloc[TR:]
y ,y_te= y[:TR],y[TR:]

ss = 0.5
mc = 0.8
md = 7
gm = 1
n_trees = 25

params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
    "num_class": num_class,
    'max_depth': md,
    'min_child_weight': mc,
    'subsample': ss,
    'colsample_bytree': 1,
    'gamma': gm,
    "eta": 0.01,
    "lambda": 0,
    'alpha': 0,
    "silent": 1,
}

dtrain = xgb.DMatrix(X, y)
dvalid = xgb.DMatrix(X_te, y_te)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
bst = xgb.train(params, dtrain, n_trees, evals=watchlist, feval=xgb_acc_score, maximize=True,
                early_stopping_rounds=esr, verbose_eval=evals)

bst.save_model(model_name)