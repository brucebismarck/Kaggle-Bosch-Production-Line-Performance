import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import gc

cat_train = pd.read_csv(".../train_categorical.csv", index_col=0, dtype= str)
cat_train_count = cat_train.count(axis = 1)
del cat_train
gc.collect()
cat_test = pd.read_csv(".../test_categorical.csv", index_col=0, dtype= str)
cat_test_count = cat_test.count(axis = 1)
del cat_test
gc.collect()
num_train = pd.read_csv('.../train_numeric.csv', index_col=0, dtype = np.float32)
num_train_count = num_train.count(axis = 1)
del num_train
gc.collect()
num_test = pd.read_csv('.../test_numeric.csv', index_col=0, dtype = np.float32)
num_test_count = num_test.count(axis = 1)
del num_test
gc.collect()
num_test_count.to_csv('.../num_test_count.csv')
num_train_count.to_csv('.../num_train_count.csv')
cat_test_count.to_csv('.../cat_test_count.csv')
cat_train_count.to_csv('.../cat_train_count.csv')
############################################################################
two_train = pd.read_csv('.../train_numeric.csv', usecols = ['Id','Response'])
two_test = pd.read_csv('.../test_numeric.csv', usecols = ['Id'])
two_test['Response'] = 0
two_train_test = pd.concat([two_train, two_test]).sort_index(by = 'Id').reset_index(drop=True)
two_train_test['if_last_1'] = 0

for i in range(two_train_test.shape[0]):
    if two_train_test.Response[i] == 1:
        two_train_test.if_last_1[i+1] = 1 
    else:
        two_train_test.if_last_1[i+1] = 0 
two_train_test = two_train_test.drop('Response',1)     
two_train_test.to_csv('.../two_train_test.csv')
############################################################################
three_train = pd.read_csv('.../train_numeric.csv', usecols = ['Id','Response'])
three_test = pd.read_csv('.../test_numeric.csv', usecols = ['Id'])
three_test['Response'] = 0
three_train_test = pd.concat([three_train, three_test]).sort_index(by = 'Id').reset_index(drop=True)
three_train_test['if_next_1'] = 0

for i in range(1,three_train_test.shape[0]):
    if three_train_test.Response[i] == 1:
        three_train_test.if_next_1[i-1] = 1 
    else:
        three_train_test.if_next_1[i-1] = 0 
three_train_test = three_train_test.drop('Response',1)     
three_train_test.to_csv('.../three_train_test.csv')
############################################################################
DATA_DIR = ".../"
ID_COLUMN = 'Id'
TARGET_COLUMN = 'Response'
SEED = 0
CHUNKSIZE = 50000
NROWS = 1183748

TRAIN_NUMERIC = "{0}/train_numeric.csv".format(DATA_DIR)
TRAIN_DATE = "{0}/train_date.csv".format(DATA_DIR)

TEST_NUMERIC = "{0}/test_numeric.csv".format(DATA_DIR)
TEST_DATE = "{0}/test_date.csv".format(DATA_DIR)

FILENAME = "etimelhoods"

train = pd.read_csv(TRAIN_NUMERIC, usecols=[ID_COLUMN, TARGET_COLUMN], nrows=NROWS)
test = pd.read_csv(TEST_NUMERIC, usecols=[ID_COLUMN], nrows=NROWS)

train["EndTime"] = -1
test["EndTime"] = -1

nrows = 0
for tr, te in zip(pd.read_csv(TRAIN_DATE, chunksize=CHUNKSIZE), pd.read_csv(TEST_DATE, chunksize=CHUNKSIZE)):
    feats = np.setdiff1d(tr.columns, [ID_COLUMN])

    stime_tr = tr[feats].max(axis=1).values
    stime_te = te[feats].max(axis=1).values

    train.loc[train.Id.isin(tr.Id), 'EndTime'] = stime_tr
    test.loc[test.Id.isin(te.Id), 'EndTime'] = stime_te

    nrows += CHUNKSIZE
    if nrows >= NROWS:
        break
    
train["StartTime"] = -1
test["StartTime"] = -1
nrows = 0
for tr, te in zip(pd.read_csv(TRAIN_DATE, chunksize=CHUNKSIZE), pd.read_csv(TEST_DATE, chunksize=CHUNKSIZE)):
    feats = np.setdiff1d(tr.columns, [ID_COLUMN])

    stime_tr = tr[feats].min(axis=1).values
    stime_te = te[feats].min(axis=1).values

    train.loc[train.Id.isin(tr.Id), 'StartTime'] = stime_tr
    test.loc[test.Id.isin(te.Id), 'StartTime'] = stime_te

    nrows += CHUNKSIZE
    if nrows >= NROWS:
        break

ntrain = train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True).reset_index(drop=False)
train_test['span'] = train_test['EndTime'] - train_test['StartTime']
train_test['f1'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
train_test['f2'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)
train_test = train_test.sort_values(by=['StartTime', 'Id'], ascending=True)
train_test['f3'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
train_test['f4'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)
train_test = train_test.sort_values(by=['index']).drop(['index'], axis=1)
train = train_test.iloc[:ntrain,:].drop(['Response'],1)
test = train_test.iloc[ntrain:, :].drop(['Response'],1)

train.to_csv(DATA_DIR+'/Farons_train_new.csv')
test.to_csv(DATA_DIR+'/Farons_test_new.csv')
####################################################################################################
#select important features from numerical
num = pd.read_csv(".../train_numeric.csv", index_col=0,
                        usecols=list(range(969)), dtype=np.float32).fillna(9999999)  
y = pd.read_csv(".../train_numeric.csv", index_col=0, usecols=[0,969], dtype=np.float32).loc[num.index].values.ravel()
X = num.values
base_score = y.sum()/y.shape[0]

clf = XGBClassifier(base_score= base_score, max_depth = 6, learning_rate = 0.05, seed = 1234)
clf.fit(X, y, eval_metric = 'auc')
#############################################################################
plt.hist(clf.feature_importances_[clf.feature_importances_>0])
important_indices_num = np.where(clf.feature_importances_>0.006)[0]
important_indices_num = pd.DataFrame(important_indices_num)
important_indices_num.to_csv('.../important_indices_num.csv')
######################################################################################
 #                      end of the numerical indices
######################################################################################
cat = pd.read_csv(".../train_categorical.csv", index_col=0, nrows = 300000, dtype= str)
cat = cat.replace(np.nan,'T0', regex=True)
cat = cat.apply(LabelEncoder().fit_transform)
y = pd.read_csv(".../train_numeric.csv", index_col=0, usecols=[0,969], dtype=np.float32).loc[cat.index].values.ravel()
clf = XGBClassifier(base_score= base_score, max_depth = 6, learning_rate = 0.05, seed = 1234)
clf.fit(cat, y , eval_metric = 'auc')
plt.hist(clf.feature_importances_[clf.feature_importances_>0])
important_indices_1 = np.where(clf.feature_importances_>0.01)[0]
######################################################################################
cat = pd.read_csv(".../train_categorical.csv", index_col=0, skiprows = 300000, nrows = 300000, dtype= str)
cat = cat.replace(np.nan,'T0', regex=True)
cat = cat.apply(LabelEncoder().fit_transform)
y = pd.read_csv(".../train_numeric.csv", index_col=0, usecols=[0,969], dtype=np.float32).loc[cat.index].values.ravel()
clf = XGBClassifier(base_score= base_score, max_depth = 6, learning_rate = 0.05, seed = 1234)
clf.fit(cat, y , eval_metric = 'auc')
plt.hist(clf.feature_importances_[clf.feature_importances_>0])
important_indices_2 = np.where(clf.feature_importances_>0.01)[0]
########################################################################################
cat = pd.read_csv(".../train_categorical.csv", index_col=0, skiprows = 600000, nrows = 300000, dtype= str)
cat = cat.replace(np.nan,'T0', regex=True)
cat = cat.apply(LabelEncoder().fit_transform)
y = pd.read_csv(".../train_numeric.csv", index_col=0, usecols=[0,969], dtype=np.float32).loc[cat.index].values.ravel()
clf = XGBClassifier(base_score= base_score, max_depth = 6, learning_rate = 0.05, seed = 1234)
clf.fit(cat, y , eval_metric = 'auc')
plt.hist(clf.feature_importances_[clf.feature_importances_>0])
important_indices_3 = np.where(clf.feature_importances_>0.01)[0]
##################################################################################
cat = pd.read_csv(".../train_categorical.csv", index_col=0, skiprows = 900000, dtype= str)
cat = cat.replace(np.nan,'T0', regex=True)
cat = cat.apply(LabelEncoder().fit_transform)
y = pd.read_csv(".../train_numeric.csv", index_col=0, usecols=[0,969], dtype=np.float32).loc[cat.index].values.ravel()
clf = XGBClassifier(base_score= base_score, max_depth = 6, learning_rate = 0.05, seed = 1234)
clf.fit(cat, y , eval_metric = 'auc')
plt.hist(clf.feature_importances_[clf.feature_importances_>0])
important_indices_4 = np.where(clf.feature_importances_>0.01)[0]
important_indices_cat = np.concatenate((important_indices_1, important_indices_2,important_indices_3,important_indices_4), axis=0)
important_indices_cat = np.unique(important_indices_cat)
pd.DataFrame(important_indices_cat).to_csv('.../important_indices_cat_0.01.csv')
##################################################################################
STATIONS = ['S32', 'S33', 'S34']
train_date_part = pd.read_csv('.../train_date.csv', nrows=10000)
date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])
date_cols = date_cols[date_cols['station'].isin(STATIONS)]
date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()
train_date = pd.read_csv('.../train_date.csv', usecols=['Id'] + date_cols)
print(train_date.columns)
train_date.columns = ['Id'] + STATIONS
for station in STATIONS:
    train_date[station] = 1 * (train_date[station] >= 0)
pattern = []
for index, row in train_date.iterrows():
    if row['S32'] == 1 and  row['S33'] == 0 and row['S34'] == 0:
        pattern.append(1)
    elif row.S32 == 1 and row.S33 == 0 and row.S34 == 1:
        pattern.append(2)
    else:
        pattern.append(0)
 
train_date['pattern'] = pattern
train_date = train_date.drop(['S33','S34'],1)

test_date = pd.read_csv('.../test_date.csv', usecols=['Id'] + date_cols)
print(test_date.columns)
test_date.columns = ['Id'] + STATIONS
for station in STATIONS:
    test_date[station] = 1 * (test_date[station] >= 0)
pattern = []
for index, row in test_date.iterrows():
    if row['S32'] == 1 and  row['S33'] == 0 and row['S34'] == 0:
        pattern.append(1)
    elif row.S32 == 1 and row.S33 == 0 and row.S34 == 1:
        pattern.append(2)
    else:
        pattern.append(0)
 
test_date['pattern'] = pattern
test_date = test_date.drop(['S33','S34'],1)

train_date.to_csv('.../train_date_feature.csv')
test_date.to_csv('.../test_date_feature.csv')
