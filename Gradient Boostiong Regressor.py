# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import ensemble

# 데이터 전처리
data = pd.read_excel('/data.xlsx')
data['distance'] = ((data['x2']-data['x1'])**2 + (data['y2']-data['y1'])**2)*100000
data = data.drop(['필요없는 컬럼명'],1)
data = data.sample(frac=1, random_state=0).reset_index(drop=True)

train = data[:12000]
test = data[12000:]
x_train = np.asarray(train.drop('label column name',1))
y_train = np.asarray(train['label column name'])
x_test = np.asarray(test.drop('label column name',1))
y_test = np.asarray(test['label column name'])

x_train = x_train / 127.5
y_train = y_train / 127.5
x_test = x_test / 127.5
y_test = y_test / 127.5

# 초기 파라미터 설정
params = {'n_estimators': 300,
          'max_depth': 3,
           'min_samples_leaf': 5,
          'learning_rate': 0.05,
          'loss': 'ls'}

# GradientBoostingRegressor 학습
reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(x_train, y_train)

# Mean Squared Error and Mean Absoluted Error 확인
mse = mean_squared_error(y_test, reg.predict(x_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The initial error of API ETA on test set: {:.4f}".format(mean_squared_error(y_test, x_test[:,0]) ))

mae = mean_absolute_error(y_test, reg.predict(x_test))
print("The mean absolute error (MAE) on test set: {:.4f}".format(mae))
print("The initial error of API ETA on test set: {:.4f}".format(mean_absolute_error(y_test, x_test[:,0]) ))

# 최적의 트리 수 찾기
reg = ensemble.GradientBoostingRegressor(**params1)
reg.fit(x_train, y_train)

errors = [mean_squared_error(y_test, y_pred) 
                             for y_pred in reg.staged_predict(x_test)]
bst_n_estimators = np.argmin(errors) + 1

reg_best = ensemble.GradientBoostingRegressor(max_depth=3, min_samples_leaf=5, learning_rate=0.01,
                                              loss='ls', n_estimators = bst_n_estimators)
reg_best.fit(x_train, y_train)

# 최적의 트리 수로 모델 학습
reg = ensemble.GradientBoostingRegressor(max_depth=3, min_samples_leaf=5, learning_rate=0.05,
                                              loss='ls', n_estimators = reg_best.n_estimators)
reg.fit(x_train, y_train)

# Mean Squared Error and Mean Absoluted Error 확인
mse = mean_squared_error(y_test, reg.predict(x_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The initial error of API ETA on test set: {:.4f}".format(mean_squared_error(y_test, x_test[:,0]) ))

mae = mean_absolute_error(y_test, reg.predict(x_test))
print("The mean absolute error (MAE) on test set: {:.4f}".format(mae))
print("The initial error of API ETA on test set: {:.4f}".format(mean_absolute_error(y_test, x_test[:,0]) ))

# Loss Graph
test_score = np.zeros((reg.n_estimators,), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(x_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(reg.n_estimators) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(reg.n_estimators) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()

# 특성 중요도 
feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure()
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(eta_features)[sorted_idx])
plt.title('Feature Importance (MDI)')
plt.show()
