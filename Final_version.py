import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import matthews_corrcoef
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import gc
gc.collect()

def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)

def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)  # number of positive
    numn = n - nump  # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    mccs = np.zeros(n)
    for i in range(n):
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
        new_mcc = mcc(tp, tn, fp, fn)
        mccs[i] = new_mcc
        if new_mcc >= best_mcc:
            best_mcc = new_mcc
            best_id = i
    if show:
        best_proba = y_prob[idx[best_id]]
        y_pred = (y_prob > best_proba).astype(int)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc

def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    return 'MCC', best_mcc

base_score = 0.00581120796927 #calculated by total 1 in response devided by length of response
##################################################################
#Feature Engineering part I, import selected features
#Raw feature selection based on basic logic, run xgboost by each num, cat, date data use params similar to final training params
#select features based on their importance, 0.06 and 0.01 are selected to balance model complexity and information input
important_indices_num = pd.read_csv('.../important_indices_num_0.06.csv') # read list of important numeric columns
important_indices_num = important_indices_num.ix[:,1]
important_indices_cat = pd.read_csv('.../important_indices_cat_0.01.csv') # read list of important categorical columns
important_indices_cat = important_indices_cat.ix[:,1]
date_columns = ['Id', 'L3_S30_D3496', 'L3_S30_D3506', 'L3_S30_D3501', 'L3_S30_D3516', 'L3_S30_D3511'] #important date columns
date_train = pd.read_csv('.../train_date.csv', usecols = date_columns) # read important date columns for training
date_test = pd.read_csv('.../test_date.csv', usecols = date_columns) # read important date columns for test
#######################################################################################
num_train = pd.read_csv(".../train_numeric.csv", index_col=0, usecols = np.concatenate([[0], important_indices_num +1]) 
                         ,dtype=np.float32).fillna(9999999)              #read important numeric training data (train)
num_test = pd.read_csv(".../test_numeric.csv", index_col=0, usecols = np.concatenate([[0], important_indices_num +1]) ,
                       dtype=np.float32).fillna(9999999)                 #read important numeric training data (test)
cat_train = pd.read_csv(".../train_categorical.csv", index_col=0, 
                        usecols= np.concatenate([[0], important_indices_cat +1]), dtype= str) #read important categorical data (train)
cat_test = pd.read_csv(".../test_categorical.csv", index_col=0,
                       usecols= np.concatenate([[0], important_indices_cat +1]), dtype= str) #read important categorical data (test)
cat_combine = pd.concat([cat_train, cat_test], axis=0,join='inner', ignore_index=False) #concat cat_train and cat_test to ensure uniform labelencoding
cat_combine = cat_combine.replace(np.nan,'T0', regex=True)  #replace all missing value via T0 which will be coverted to 0 in labelencoder
cat_combine = cat_combine.apply(LabelEncoder().fit_transform)
cat_train = cat_combine[0:1183747] # seperate 
cat_test = cat_combine[1183747:] 
#########################################################################################
# Feature Engineering part II import created features
#########################################################################################
foran_train = pd.read_csv(".../Farons_train_new.csv",index_col=0) # foran's feature https://www.kaggle.com/mmueller/bosch-production-line-performance/road-2-0-4
foran_test = pd.read_csv(".../Farons_test_new.csv",index_col=0) # I added some related features like endtime, timespan etc.

two_train_test = pd.read_csv('.../two_train_test.csv' , index_col = 0) # this and next features are designed to show if product i is failure product,
three_train_test = pd.read_csv('.../three_train_test.csv' , index_col = 0) # then, i-1 and i+1 have higher probability to be a failure product
# idea of these two features are generated from this discussion https://www.kaggle.com/aamaia/bosch-production-line-performance/failures-in-pairs

S32andpattern_train = pd.read_csv('.../train_date_feature.csv', index_col = 0) #this feature generated from https://www.kaggle.com/gaborfodor/bosch-production-line-performance/69-failure-rate
S32andpattern_test = pd.read_csv('.../test_date_feature.csv', index_col = 0) # dummy features
S32andpattern_train = S32andpattern_train.merge(pd.get_dummies(S32andpattern_train.pattern), left_index = True, right_index = True, how = 'inner').drop('pattern',1)
S32andpattern_test = S32andpattern_test.merge(pd.get_dummies(S32andpattern_test.pattern), left_index = True, right_index = True, how = 'inner').drop('pattern',1)

col47_train = pd.read_csv('.../train_date.csv', usecols = ['Id','L3_S47_D4150']).fillna(0) # following four features 
col47_test = pd.read_csv('.../test_date.csv', usecols = ['Id','L3_S47_D4150']).fillna(0) # col47 col0 col13 and col38 
col0_train = pd.read_csv('.../train_date.csv', usecols = ['Id','L0_S0_D1']).fillna(0) # ignited by https://www.kaggle.com/gingerman/ 
col0_test = pd.read_csv('.../test_date.csv', usecols = ['Id','L0_S0_D1']).fillna(0) #bosch-production-line-performance/shopfloor-visualization-2-0
col13_train = pd.read_csv('.../train_date.csv', usecols = ['Id','L0_S13_D355']).fillna(0) #production through different paths 
col13_test = pd.read_csv('.../test_date.csv', usecols = ['Id','L0_S13_D355']).fillna(0) # may have different failure rate
col38_train = pd.read_csv('.../train_date.csv', usecols = ['Id','L3_S38_D3953']).fillna(0) # this is easy to understand
col38_test = pd.read_csv('.../test_date.csv', usecols = ['Id','L3_S38_D3953']).fillna(0)
col47_train['L3_S47_D4150']  = np.where(col47_train['L3_S47_D4150']>0, 1, 0) #convert date data to 01 data
col0_train['L0_S0_D1']       = np.where(col0_train['L0_S0_D1']>0, 1, 0)
col13_train['L0_S13_D355']   = np.where(col13_train['L0_S13_D355']>0, 1, 0)
col38_train['L3_S38_D3953']  = np.where(col38_train['L3_S38_D3953']>0, 1, 0)
col47_test['L3_S47_D4150']  = np.where(col47_test['L3_S47_D4150']>0, 1, 0)
col0_test['L0_S0_D1']       = np.where(col0_test['L0_S0_D1']>0, 1, 0)
col13_test['L0_S13_D355']   = np.where(col13_test['L0_S13_D355']>0, 1, 0)
col38_test['L3_S38_D3953']  = np.where(col38_test['L3_S38_D3953']>0, 1, 0)

num_train_count = pd.read_csv('.../num_train_count',header = 0) # how many numeric features measured of each product 
num_test_count = pd.read_csv('.../num_test_count',header = 0)
cat_train_count = pd.read_csv('.../cat_train_count',header = 0) #how many categorical features measured of each product
cat_test_count = pd.read_csv('.../cat_test_count',header = 0)
train_count = num_train_count.merge(cat_train_count, left_on = 'Id', right_on = 'Id', how = 'inner')
test_count = num_test_count.merge(cat_test_count, left_on = 'Id', right_on = 'Id', how = 'inner')
train_count['count_sum'] = train_count['count_x'] + train_count['count_y'] # in total, how many features measured
test_count['count_sum'] = test_count['count_x'] + test_count['count_y']

y = pd.read_csv(".../train_numeric.csv", index_col=0, dtype=np.float32, usecols=[0,969]) #read response
X = num_train.merge(cat_train, left_index = True,right_index = True, how='inner') # mearge up all features
X = X.merge(foran_train, left_index = True, right_on = 'Id', how = 'inner') 
X = X.merge(date_train, left_on = 'Id', right_on = 'Id', how ='inner')
X = X.merge(S32andpattern_train, left_on = 'Id', right_on = 'Id', how = 'inner')
X = X.merge(two_train_test, left_on = 'Id', right_on = 'Id', how = 'inner')
X = X.merge(three_train_test, left_on = 'Id', right_on = 'Id', how = 'inner' )
X = X.merge(col47_train, left_on = 'Id', right_on = 'Id', how = 'inner' )
X = X.merge(col0_train, left_on = 'Id', right_on = 'Id', how = 'inner' )
X = X.merge(col13_train, left_on = 'Id', right_on = 'Id', how = 'inner' )
X = X.merge(col38_train, left_on = 'Id', right_on = 'Id', how = 'inner' )
X = X.merge(train_count, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_y_whole = xgboost.DMatrix(X,y) #XGBOOST need convert data to DMatrix form

X_test_whole = num_test.merge(cat_test, left_index = True, right_index = True, how='inner') #do same thing for test data
X_test_whole = X_test_whole.merge(foran_test, left_index = True, right_on = 'Id', how = 'inner')
X_test_whole = X_test_whole.merge(date_test, left_on = 'Id', right_on = 'Id', how ='inner')
X_test_whole = X_test_whole.merge(S32andpattern_test, left_on = 'Id', right_on = 'Id', how = 'inner')
X_test_whole = X_test_whole.merge(two_train_test, left_on = 'Id', right_on = 'Id', how = 'inner')
X_test_whole = X_test_whole.merge(three_train_test, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_test_whole = X_test_whole.merge(col47_test, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_test_whole = X_test_whole.merge(col0_test, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_test_whole = X_test_whole.merge(col13_test, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_test_whole = X_test_whole.merge(col38_test, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_test_whole = X_test_whole.merge(test_count, left_on = 'Id', right_on = 'Id', how = 'inner' )
X_test_whole = xgboost.DMatrix(X_test_whole)  

X_y = X.merge(y, left_on = 'Id', right_index = True, how='inner' )
X_y_train, X_y_test = train_test_split(X_y, test_size = 0.2)
X_train = X_y_train.iloc[:,:-1]
y_train = X_y_train.iloc[:,-1]
X_test = X_y_test.iloc[:,:-1]
y_test = X_y_test.iloc[:,-1]
X_test_1 = xgboost.DMatrix(X_test, y_test)
X_test_2 = xgboost.DMatrix(X_test)
X_train = xgboost.DMatrix(X_train, y_train)
watchlist = [(X_train,'train'),(X_test_1,'val')]
params = {
    'booster':'gbtree',
    'objective' : 'reg:logistic',
    'colsample_bytree' : 0.2,
    'min_child_weight':4,
    'subsample' : 1,
    'learning_rate': 0.1,
    'max_depth':6,
    'gamma': 0.05,
    'seeds' : 1234
}
model_test = xgboost.train(params, X_y_whole, 1000,  watchlist, feval=mcc_eval, early_stopping_rounds=60,  maximize=True) 
# from X_y merge to here is used to fast determine round number

#Use cross validation will get precise round number
cv = xgboost.cv(params, X_y_whole, 1000, nfold = 5, stratified=True, maximize=True )   
cv  
model = xgboost.train(params, X_y_whole, 230,  feval=mcc_eval,  maximize=True) #230 is the cv round, maybe changed
# can use rf to make ensemble. Because I dont have enough time and time of trying, I didn't use it in my final submission
#rfmodel = rf(n_estimators = 250, max_depth = 20, n_jobs = 7, oob_score = True,   max_features = 10)
#X_nonan = X.fillna(9999999)
#rfmodel.fit(X_nonan,y)
#X_test_whole_nonan = X_test_whole.fillna(9999999)
#y_rf = rfmodel.predict(X_test_whole_nonan)
#y_rf = np.int32(y_rf)
#np.sum(y_pred)

# following 7 lines could be used to determine 
train_prob_predict = model.predict(X_test_2)
thresholds = np.linspace(0.2, 0.5, 1000)
mcc_1 = np.array([matthews_corrcoef(y_test, train_prob_predict>thr) for thr in thresholds])
plt.plot(thresholds, mcc_1)
best_threshold = thresholds[mcc_1.argmax()]
print(mcc_1.max())
print(best_threshold)

predictions_real = model.predict(X_test_whole)  
y_pred = (predictions_real > 0.365).astype(int) #by control threshold value to find the best one
sub = pd.read_csv(".../sample_submission.csv", index_col=0)
sub["Response"] = y_pred
sub.to_csv(".../final_submission.csv")
