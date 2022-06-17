# 使用套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder as LE, OneHotEncoder as OHE, PolynomialFeatures

# note:
# 這支程式的目的，是為了展示多項式迴歸和線性迴歸在非線性資料的差異度。

# 1. 建立資料(非線性)
## 1.1 建立座標軸
data = [ [1, 1], [1.2, 3], [1.5, 7], [1.6, 10], [1.7, 11], [1.8, 11], [1.85, 13], [1.9, 15], [3.0, 17], [4.2, 18], [4.5, 19], [5, 19.5], [6, 19.6], [7, 19.65], [8, 19.8], [10, 20], [11, 20.2]]
data = np.array(data)
## 1.2 將坐標軸處理成特徵x與標籤y
x = [ [k[0]] for k in data ]
y = [ k[1] for k in data ]
plt.figure(figsize=(10, 4))
plt.title('It is nonlinear data.')
plt.scatter(x,y)
plt.show()


# 2. 線性迴歸以及多項式迴歸的比較
## 2.1 線性迴歸
## 建立 LR, 訓練及預測
regs = LR()
regs.fit(x, y)
result = regs.predict(x)

## 畫圖 (預測趨勢圖)
plt.figure(figsize=(10, 4))
plt.scatter(x,y)
plt.plot(x, result, color='red')
plt.title("linear")
plt.show()

## 2.2 多項式迴歸
## polynomial process
## 定義一個能將原始資料轉換成指數資料的 function
pole = PolynomialFeatures(degree=3)
PR_data = pole.fit_transform(x)
## 建立 LR, 訓練及預測
regs2 = LR()
regs.fit(PR_data, y)
result2 = regs.predict(PR_data)
## 畫圖 (預測趨勢圖)
plt.figure(figsize=(10, 4))
plt.scatter(x,y)
plt.plot(x, result2, color='red')
plt.title("polynomial")
plt.show()
