# 使用套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import LabelEncoder as LE, OneHotEncoder as OHE, PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 1. 資料前處理
## 1.1 資料讀取
## (這邊將前一天收盤價視為data， 當天則為預測值)
df = pd.read_csv('./data/TSM.csv')
dates = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='ignore')
data = df.iloc[:-1,1:].values #和線性迴歸的差異 (特徵值不只有一種)
label = df.iloc[1:,4].values

## 1.2 資料切割 (分成訓練集和測試集)
X_train = data[:-100]
X_test = data[-100:]
y_train = label[:-100]
y_test = label[-100:]
y_dates = dates[-100:]

## 1.3 將原始資料轉換成指數資料
## polynomial process
## 定義一個能將原始資料轉換成指數資料的 function
pole = PolynomialFeatures(degree=2)
PR_data = pole.fit_transform(X_train)

# 2. 建立模型, 模型訓練, 模型預測
## 2.1 使用原始資料做訓練
regs = LR()
regs.fit(X_train, y_train)
result = regs.predict(X_test)

## 2.2 使用指數資料做訓練(多項式迴歸)
regs2 = LR()
regs2.fit(PR_data, y_train)
PR_X_test = pole.fit_transform(X_test)
result2 = regs2.predict(PR_X_test)

## 2.3 畫圖 (預測股價的趨勢圖)
plt.figure(figsize=(16, 4))
dx = range(0, y_test.shape[0])
dx = y_dates
plt.plot(dx, y_test, color='blue', label='real')
plt.plot(dx, result, color='red', label='MLR_predict')
plt.plot(dx, result2, color='green', label='polyR_predict')
plt.legend()
plt.show()


# 3. 評估
# 自己寫算法
def MSE(y, y_pred):
  mse = np.mean((y - y_pred)**2) 
  return mse
mlr_MSE = MSE(y_test, result)
poly_MSE = MSE(y_test, result2)
print('mlr_test_mse:', mlr_MSE)
print('poly_test_mse:', poly_MSE)

# 使用 sklearn套件
mlr_mse = mean_squared_error(y_test, result, squared=True)
poly_mse = mean_squared_error(y_test, result2, squared=True)
print('mlr_test_mse:', mlr_mse)
print('poly_test_mse:', poly_mse)









