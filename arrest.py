# 日付map
def date(x):
    b=int(str(x).split('-')[1])
    if  str(x).split('-')[0]=='2013':
        a=0
    elif str(x).split('-')[0]=='2014':
        a=12
    elif str(x).split('-')[0]=='2015':
        a=24
    return a+b

# 時間map
def time(x):
    return str(x).split(':')[0]

# 年齢map
def age(x):
    return  str(int(int(x)/10))
		
# 文字列変換   
def toStr(x):
    return str(x)

# 文字列を分離して個数をリターン		
def splitStr(x):
    return len(str(x).split(','))




# 以下前処理

# 以下のモジュールを使うので、あらかじめ読み込んでおいてください
import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame
# 可視化モジュール
import matplotlib.pyplot as plt
import matplotlib as mpl

import math
import sys
sys.setrecursionlimit(220000)

officer_del=[]
location_del=[]
def preprocess(arr,isTrain):
    global officer_del
    global location_del
    
    #state
    arr=arr.drop('state',axis=1)
    
    #date
    arr['stop_date'] = arr['stop_date'].map(date)
    
    A=pd.get_dummies(arr[['stop_date']].astype(object))
    arr=pd.concat([arr, A], axis=1)
    arr=arr.drop('stop_date',axis=1)
    
    #time
    arr['stop_time'] = arr['stop_time'].map(time)

    B=pd.get_dummies(arr[['stop_time']])
    arr=pd.concat([arr, B], axis=1)
    arr=arr.drop('stop_time',axis=1)
    arr=arr.drop('stop_time_nan',axis=1)
    
    #location
    C=pd.get_dummies(arr[['location_raw']])
    C=C.sort_index()
    C_idx = C.shape[0]
    C_col = C.shape[1]-1
    C['location_raw_others']=np.zeros(C_idx)
    ratio_border=0.036
    if isTrain: 
        location_ratio=arr.groupby('location_raw').sum().is_arrested/arr['location_raw'].value_counts().sort_index()
        location_del = [C.columns[i] for i in range(C_col) if location_ratio[i]<ratio_border]
        for i in range(C_col) :
            if location_ratio[i]<ratio_border:
                C['location_raw_others'] = C.iloc[:,i] + C['location_raw_others']

    else :
        src_set = set(location_del)
        tag_set = set(C.columns)
        matched_list = list(src_set & tag_set)
        for i in range(len(matched_list)):
            C['location_raw_others'] = C[matched_list[i]] + C['location_raw_others']


    src_set = set(location_del)
    tag_set = set(C.columns)
    matched_list = list(src_set & tag_set)
    C=C.drop(columns=matched_list)

    arr=pd.concat([arr, C], axis=1)
    arr=arr.drop('location_raw',axis=1)
    
    #county
    arr=arr.drop('county_name',axis=1)
    arr=arr.replace({9001.0: '0', 9003.0: '1', 9005.0: '2', 9007.0: '3', 9009.0: '4', 9011.0: '5', 9013.0: '6', 9015.0: '7'})
    E=pd.get_dummies(arr[['county_fips']])
    arr=pd.concat([arr, E], axis=1)
    arr=arr.drop('county_fips',axis=1)
    
    #grain_loc
    arr=arr.drop('fine_grained_location',axis=1)
    
    #police_dep
    arr=arr.drop('police_department',axis=1)
    
    #gender
    F=pd.get_dummies(arr[['driver_gender']])
    arr=pd.concat([arr, F], axis=1)
    arr=arr.drop('driver_gender',axis=1)
    
    #age
    arr['driver_age']=arr['driver_age'].fillna(0)

    arr['driver_age']=arr['driver_age'].map(age)
    
    G=pd.get_dummies(arr[['driver_age']].astype(object))
    arr=pd.concat([arr, G], axis=1)
    
    arr=arr.drop('driver_age',axis=1)
    arr=arr.drop('driver_age_raw',axis=1)
    
    #race
    H=pd.get_dummies(arr[['driver_race']])
    arr=pd.concat([arr, H], axis=1)
    arr=arr.drop('driver_race',axis=1)
    arr=arr.drop('driver_race_raw',axis=1)
    
    #violation
    arr['violation_num']=np.zeros(arr.shape[0])
    
    arr['violation_num'] = arr['violation'].map(splitStr)

    violation_first = [ (str(x).split(',')[0]) for x in arr['violation'] ]
    violation_first_dm =pd.get_dummies(violation_first)

    for i in range(arr.shape[0]):
        if arr['violation_num'][i]==2:
            violation_first_dm[arr['violation'].values[i].split(',')[1]].values[i] += 1
        elif arr['violation_num'][i]==3:
            violation_first_dm[arr['violation'].values[i].split(',')[1]].values[i] += 1
            violation_first_dm[arr['violation'].values[i].split(',')[2]].values[i] += 1
        elif arr['violation_num'][i]==4:
            violation_first_dm[arr['violation'].values[i].split(',')[1]].values[i] += 1
            violation_first_dm[arr['violation'].values[i].split(',')[2]].values[i] += 1
            violation_first_dm[arr['violation'].values[i].split(',')[3]].values[i] += 1
        elif arr['violation_num'][i]==5:
            violation_first_dm[arr['violation'].values[i].split(',')[1]].values[i] += 1
            violation_first_dm[arr['violation'].values[i].split(',')[2]].values[i] += 1
            violation_first_dm[arr['violation'].values[i].split(',')[3]].values[i] += 1
            violation_first_dm[arr['violation'].values[i].split(',')[4]].values[i] += 1

    arr=pd.concat([arr, violation_first_dm], axis=1)
    arr=arr.drop('violation',axis=1)
    arr=arr.drop('violation_raw',axis=1)
    
    #search-sconducted
    I=pd.get_dummies(arr[['search_conducted']].astype(object))
    arr=pd.concat([arr, I], axis=1)
    arr=arr.drop('search_conducted',axis=1)
    
    #search-type
    J=pd.get_dummies(arr[['search_type']].astype(object),dummy_na=True)
    arr=pd.concat([arr, I], axis=1)
    arr=arr.drop('search_type',axis=1)
    arr=arr.drop('search_type_raw',axis=1)
    
    #contraband
    K=pd.get_dummies(arr[['contraband_found']].astype(object))
    arr=pd.concat([arr, K], axis=1)
    arr=arr.drop('contraband_found',axis=1)
    
    #officer
    arr['officer_id'] = arr['officer_id'].map(toStr)
    L=pd.get_dummies(arr[['officer_id']].astype(object))
    L=L.sort_index()
    L_idx = L.shape[0]
    L_col = L.shape[1]
    L['officer_others']=np.zeros(L_idx)
    officer_border=5
    if isTrain: 
        officer_search_cnt=arr.groupby('officer_id')['is_arrested'].count()
        officer_del = [L.columns[i] for i in range(L_col) if officer_search_cnt[i]<officer_border]
        for i in range(L_col) :
            if officer_search_cnt[i]<officer_border:
                L['officer_others'] = L.iloc[:,i] + L['officer_others']
    
    else :
        off_set = set(officer_del)
        L_set = set(L.columns)
        matched_list = list(off_set & L_set)
        for i in range(len(matched_list)):
            L['officer_others'] = L[matched_list[i]] + L['officer_others']
        

    off_set = set(officer_del)
    L_set = set(L.columns)
    matched_list = list(off_set & L_set)
    L=L.drop(columns=matched_list)

    arr=pd.concat([arr, L], axis=1)
    arr=arr.drop('officer_id',axis=1)
    
    #duration
    duration_mapping = {'1-15 min':1  , '16-30 min':2, '30+ min':3}
    arr['stop_duration']=arr['stop_duration'].map(duration_mapping)
#     M=pd.get_dummies(arr[['stop_duration']].astype(object))
#     arr=pd.concat([arr, M], axis=1)
#     arr=arr.drop('stop_duration',axis=1)
    
    return arr


# テストとトレインに分離
def Split(arr):
    from sklearn.model_selection import train_test_split, GridSearchCV
    X ,y =arr.drop('is_arrested',axis=1) , arr['is_arrested']
    #列名順に並び替え
    X=X.sort_index(axis=1, ascending=True)
    X_train ,X_test, y_train, y_test =  train_test_split(X, y , test_size=0.2 , random_state=0, stratify=y)
    return X_train ,X_test, y_train, y_test

# 標準化
def Standardize(X_train):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    #fitで、平均値と標準偏差を推定
    sc.fit(X_train)
    #transformで、fitで推定した値を元に標準化
    X_train_std = sc.transform(X_train)
    X_test_std  = sc.transform(X_test)
    return X_train_std,X_test_std 

# ランダムフォレスト
def forest_fit(X_train_std, y_train):
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(criterion='gini', n_estimators=30, random_state=1,n_jobs=2)
    forest.fit(X_train_std,y_train)
    return forest

# ロジスティック回帰
def lr_fit(X_train_std, y_train):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l1', C=1)
    lr.fit(X_train_std, y_train)
    return lr

# 交差検証
def cross_val(method,X_train_std, y_train):
    from sklearn.model_selection import cross_val_score
    # 交差検証
    scores = cross_val_score(method, X_train_std, y_train, cv=5)
    # 各分割におけるスコア
    print('Cross-Validation scores: {}'.format(scores))
    # スコアの平均値
    import numpy as np
    print('Average score: {}'.format(np.mean(scores)))

#　AUC評価
def AUC(method ,X_test_std, y_test):
    from sklearn import metrics
    from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
    print(roc_auc_score(y_test, method.predict_proba(X_test_std)[:,1]))




### 学習
arr=preprocess(arr,True)
X_train,X_test,y_train, y_test =Split(arr)
X_train_std, X_test_std=Standardize(X_train)
from sklearn.ensemble import GradientBoostingRegressor

x = X_train_std
y = y_train
# 勾配ブースティング木
model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1, 
              max_depth= 4,
              min_samples_leaf=5,
              max_features=0.1)
model.fit(x, y)

def bord(x):
    if x<0 :
        return 0.0
    elif x>1:
        return 1.0
    else:
        return x
A=pd.DataFrame(model.predict(X_test_std))
A[0]=A[0].map(bord)
print(roc_auc_score(y_test, A[0]))


###　予測

test = pd.read_csv('test.csv')
test=preprocess(test, False)

#testカラムで必要なものを追加、不要なものを削除
src_set = set(test.columns)
tag_set = set(X_train.columns)

if list(set(tag_set) -set(src_set & tag_set)) != []:
    train_margin = list(set(tag_set) -set(src_set & tag_set))
    for i in range(len(train_margin)):
        test[train_margin[i]]=np.zeros(test.shape[0])
        
if list(set(src_set) -set(src_set & tag_set)) != [] :
    test_margin = list(set(src_set) -set(src_set & tag_set))
    test=test.drop(columns =test_margin)

test=test.sort_index(axis=1)

#transformで、fitで推定した値を元に標準化
test_std = sc.transform(test)

ans = pd.DataFrame(lr4.predict_proba(test_std)[:,1])
ans.columns=["is_arrested"]
ans['is_arrested']=ans['is_arrested'].map(bord)
ans = ans.round(6)
ans.to_csv('submit.csv',index=False)
