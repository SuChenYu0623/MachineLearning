# 使用套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error

# 1. 資料前處理
## 1.1 資料讀取
## (這邊將前一天收盤價視為data， 當天則為預測值)
df = pd.read_csv('TSM.csv')
dates = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='ignore')
data = df.iloc[:-1,[1]].values #這裡是以開盤價
label = df.iloc[1:,4].values

## 1.2 資料切割 (分成訓練集和測試集)
X_train = data[:-100]
X_test = data[-100:]
y_train = label[:-100]
y_test = label[-100:]
y_dates = dates[-100:]


# 2. 建立模型, 模型訓練, 模型預測
## 2.1 建立 LR, 訓練及預測
regs = LR()
regs.fit(X_train, y_train)
result = regs.predict(X_test) 

## 2.2 畫圖 (預測股價的趨勢圖)
plt.figure(figsize=(10, 4))
dx = range(0, y_test.shape[0])
dx = y_dates
plt.plot(dx, y_test, color='blue', label='real')
plt.plot(dx, result, color='red', label='predict')
plt.title('predict')
plt.legend()
plt.show()

## 2.3 評估模型
# 自己寫算法
def MSE(y, y_pred):
  return np.mean((y - y_pred)**2) 
MSE = MSE(y_test, result)
print('test_MSE', MSE)
# 使用 sklearn套件
mse = mean_squared_error(y_test, result, squared=True)
print('test_mse:', mse)


# 3. 了解測試集和預測值之間的線性關係
## 3.1 線性回歸的意義
## 不同的x對應不同的y，找出一條線性的最佳解
## 但一般來說不會有線性問題可以解決，因為實務上很少問題是單純的線性關係
plt.figure(figsize=(10, 4))
plt.scatter(X_test, y_test, color='blue', label="source data")
plt.plot(X_test, result, color='red', label="object line")
plt.legend()
plt.show()

# 4. 計算 y = aX+b 的 a 和 b
xmean = np.mean(X_train)
ymean = np.mean(y_train)

tmp1, tmp2 = 0.0, 0.0
for i, j in zip(X_train, y_train):
    tmp1 += (i-xmean)*(j-ymean)
    tmp2 += (i-xmean)**2
a = tmp1/tmp2
b = ymean - a*(xmean)
print('a:', a, 'b:', b)
print('sklearn:', result[0])
print('math formulation:', a*X_test[0]+b)

