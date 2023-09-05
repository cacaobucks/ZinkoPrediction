# 必要なライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
%matplotlib inline

# 人口構成のCSVファイルの読み込み
df = pd.read_csv("population_data/japan_population.csv")
df.head()

# 全件数の確認
df.shape

#読み込んだCSVデータから、東京都のものだけを取り出す
tokyo = df[df["都道府県名"] == "東京都"]
tokyo.head(10)

# 1960年から2015年までの東京都の総人口数データを機械学習にかける。
# ある年の総人口数を説明変数X、その翌年の総人口数を目的変数yに設定する
# （Xは縦56行・横1列の二次元配列、yは長さ56の一次元配列としてndarrayを作成する）
X = np.empty((56, 1), dtype=np.uint32)
y = np.empty(56 , dtype=np.uint32)

# 人口はデータの左から3番目
for i in range(56):
    X[i][0] = tokyo.iloc[i, 3]
    y[i] = tokyo.iloc[i + 1, 3]
    
#X ,ｙの値を、簡単に先頭10件ほどで確認
print(X[0:10])
print(y[0:10])


# 1960年から2009年までを訓練データ、
# 2010年以降をテストデータとして分割する
X_train = X[:50]
X_test = X[50:]
y_train = y[:50]
y_test = y[50:]


# 線形回帰モデルの作成と学習の実行
model = LinearRegression()
model.fit(X_train, y_train)

# テストデータで「翌年の総人口」予測の実施
y_pred = model.predict(X_test)

# 予測結果は実数値のため、整数値に変換
y_pred = y_pred.astype(np.uint32)
print(y_pred)


#CSVに記載されている正解値は、y_test に読み込んでありますから、それを表示して比較
print(y_test)


# 正解値とグラフで比較するため
# 実測値と予測値とを連結させた配列y_pred_grを作成
y_pred_gr = np.concatenate([y_train, y_pred])


# 正解値と予測値のグラフ表示
plt.plot(range(56), y_pred_gr, label='Predicted', color='red')
plt.plot(range(56), y, label='Actual', color='blue')
plt.xlabel('Years')
plt.ylabel('Population')
plt.title("Tokyo's population")
plt.grid(True)
plt.legend(loc = "upper left")
