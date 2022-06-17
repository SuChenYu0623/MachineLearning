# 使用套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR, LogisticRegression as LogR
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. 資料前處理 (因為 SVM 是分類問題，所需要將標籤處理成兩類，漲與跌)
## 1.1 資料讀取與建立新標籤
df = pd.read_csv('./data/TSM.csv')
dates = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='ignore')
data = df.iloc[:-1,1:].values
stock_prices = df.iloc[:,4].values # 真實股價
## 產生對應的label
label = df.iloc[:,4].values
label = [ 1 if k>label[idx-1] else 0 for idx, k in enumerate(label[1:])]
label = np.array(label)
print(data.shape)
print(label.shape)

## 1.2 資料切割
X_train = data[:-100]
X_test = data[-100:]
y_train = label[:-100]
y_test = label[-100:]
test_dates = dates[-100:]
test_stock_prices = stock_prices[-100:]

## 1.3 正規化
#StandardScaler (平均值和標準差)
#MinMaxScaler(最小最大值標準化)
#MaxAbsScaler（絕對值最大標準化）
#RobustScaler
fS = StandardScaler() #平均值和標準差
X_train = fS.fit_transform(X_train)
X_test = fS.fit_transform(X_test)

# 2. 建立模型、模型訓練、模型預測
cls = SVC(kernel='linear', random_state=0)
cls.fit(X_train, y_train)
pred = cls.predict(X_test)

# 3. 評估模型準確率
count = 0
for i in (pred==y_test):
    if i==True:
        count += 1
print('acc:', count/len(y_test))


