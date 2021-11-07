import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib inline

import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

# 파일 불러오기
dir = '..file name.xlsx'
tada_eta = pd.read_excel(dir)
print(tada_eta.columns)
tada_eta.tail()

# 이동 거리 구하고 불필요한 feature 삭제
distance = np.sqrt(((tada_eta['pickup_lat']-tada_eta['driver_lat'])**2 + (tada_eta['pickup_lng']-tada_eta['driver_lng'])**2))
tada_eta['distance'] = distance
tada_eta = tada_eta.drop(['id', 'created_at_kst', 'driver_id', 'pickup_lng', 'pickup_lat', 'driver_lng','driver_lat'],axis=1)
tada_eta.head()

# 서울시 공공 데이터셋을 사용하여 월별로 서울시 자치구의 차량 통행 속도 데이터를 이용하기 위해 데이터를 불러왔습니다.
speed_month = pd.read_excel('./서울시 차량 통행 속도.xlsx')
speed_month = speed_month.iloc[:, 2:].drop(['평균'], axis=1)
speed_month.columns = ['자치구', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
speed_month = speed_month.set_index('자치구')
speed_month = speed_month.drop(['전체', '강남', '강북'], axis=0)
speed_month.head()

# tada_eta의 pickup_gu 컬럼과 month 컬럼을 이용해
# speed_month 데이터와 일치하는 데이터를 불러와 tada_eta 데이터에 speed 라는 컬럼을 만들어 주었습니다.

con1 = tada_eta['pickup_gu']
con2 = tada_eta['month']
tada_eta['speed'] = [speed_month.loc[con1[i], con2[i]] for i in range(len(tada_eta))]
tada_eta.head()

# 데이터 순서 섞기
tada_eta = tada_eta.sample(frac=1, random_state=0).reset_index(drop=True)
tada_eta.head()

# Ordianl Encoding
# pickup_gu 컬럼을 이용하기 위해 숫자 형태로 인코딩
enc = OrdinalEncoder(dtype=np.int32)
ordinal = enc.fit_transform(np.asarray(tada_eta['pickup_gu']).reshape(-1,1))
tada_eta['pickup_gu'] = ordinal[:,0]
tada_eta.head()

# 데이터 확인
tada_eta.describe()

# trainset과 testset 분류
train = tada_eta[:12000]
test = tada_eta[12000:]

xtrain = np.asarray(train.drop(['ATA', 'pickup_gu'], 1))
ytrain = np.asarray(train['ATA'])

xtest = np.asarray(test.drop(['ATA', 'pickup_gu'], 1))
ytest = np.asarray(test['ATA'])

train_gu = np.asarray(train[['pickup_gu']]).reshape(-1, 1)
test_gu = np.asarray(test[['pickup_gu']]).reshape(-1, 1)

(xtrain.shape, ytrain.shape), (xtest.shape, ytest.shape), (train_gu.shape, test_gu.shape)

# PCA를 이용해 보고자 했으나 feature가 몇 안되기 때문에 성능이 눈에 띄게 향상되지 않는 관계로 생략
# 각기 다른 데이터의 범위를 맞춰주기 위해 표준화 진행
scaler = StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# pca = PCA(n_components=4)
# pca.fit(xtrain)
# xtrain = pca.transform(xtrain)
# xtest = pca.transform(xtest)

xtrain = np.concatenate((xtrain, train_gu), axis=1)
xtest = np.concatenate((xtest, test_gu), axis=1)

xtrain.shape, xtest.shape

# 인코딩한 컬럼을 이용하기 위해 HistGradientBoostingRegressor를 사용해 주었습니다.
reg = sklearn.ensemble.HistGradientBoostingRegressor(
    categorical_features = [5],
    early_stopping = False,
    random_state=0,
    
    max_iter=500,
    learning_rate = 0.06,
    max_depth=1,
    min_samples_leaf = 1,
    max_bins=100,
    loss='squared_error',
    scoring='loss',

)
reg.fit(xtrain, ytrain)
mse = mean_squared_error(ytest, reg.predict(xtest))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

mae = mean_absolute_error(ytest, reg.predict(xtest))
print("The mean absolute error (MAE) on test set: {:.4f}".format(mae))

train_score = np.zeros((reg.max_iter,), dtype=np.float64)
test_score = np.zeros((reg.max_iter,), dtype=np.float64)

# 그래프를 통해 Loss 확인
for i, y_pred in enumerate(reg.staged_predict(xtrain)):
    train_score[i] = mean_squared_error(ytrain, y_pred)

for i, y_pred in enumerate(reg.staged_predict(xtest)):
    test_score[i] = mean_squared_error(ytest, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(reg.max_iter) + 1, train_score, 'b-', label='Training Set Deviance')
plt.plot(np.arange(reg.max_iter) + 1, test_score, 'r-', label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()

