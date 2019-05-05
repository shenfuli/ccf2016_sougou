'''concat learn data and test data'''

import pandas as pd
import cfg

df_tr = []
for i, line in enumerate(open(cfg.data_path + 'user_tag_query.10W.TRAIN')):
    segs = line.split('\t')
    row = {}
    row['Id'] = segs[0]
    row['age'] = int(segs[1])
    row['gender'] = int(segs[2])
    row['Education'] = int(segs[3])
    row['query'] = '\t'.join(segs[4:])
    df_tr.append(row)

df_tr = pd.DataFrame(df_tr, columns=['Id', 'age', 'gender', 'Education', 'query'])
print(df_tr.shape)

print(df_tr['age'].value_counts())
print(df_tr['gender'].value_counts())
print(df_tr['Education'].value_counts())

df_all = df_tr
df_all.to_csv(cfg.data_path + 'all_v2.csv', index=None)
