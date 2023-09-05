# 必要なライブラリのインポート
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# 人口構成のCSVファイルの読み込み
df = pd.read_csv("population_data/japan_population.csv")
df.head()


# 1960年から2015年までの関東地方各県の総人口数データを機械学習にかける。
# ある年の都道府県コードをカテゴリ値 とした上で ダミー変数化したものと総人口数を説明変数X、その翌年の総人口数を目的変数yに設定する
# （Xは縦56×7行の横7列の二次元配列、yは縦56×7の長さの1次元配列としてndarrayを作成する）
X = np.zeros((56 * 7,7), dtype=np.uint32)
y = np.zeros(56 * 7, dtype=np.uint32)

cnt = 0
for i in range(56 * 47):
    pref_id = df.iloc[i, 1]
    population = df.iloc[i, 3]
    next_population = df.iloc[i + 47, 3]

    if pref_id >= 8 and pref_id <= 14:    #pref_id 8(茨城県)から、pref_id 14(神奈川県)を計測データXに格納する。  
        if pref_id < 14:                  #pref_id 14(神奈川県)は、ダミー変数化の際に1としない。 
            X[cnt][pref_id - 8] = 1       #計測データXに都県のダミーデータを格納する。値は1とする。8~13のpref_idを0~5として番号を振りなおす。  

        X[cnt][6] = population            #計測データXの7番目の要素は、その年の人口とする。  
        y[cnt] = next_population          #教師データyは、翌年の人口とする。  
        cnt += 1                          #カウンタ変数cntをカウントアップする。  

# Xを確認
print(X[0:10])


# yを確認
print(y[0:10])


# 1960年から2009年までを訓練データ、
# 2010年以降のデータをテストデータとして分割する
X_train = X[:350]
X_test = X[350:]
y_train = y[:350]
y_test = y[350:]


# LinearRegressionで回帰モデルを作成
model1 = LinearRegression()
model1.fit(X_train, y_train)


# LinearRegressionで予測実行
y_pred1 = model1.predict(X_test)

# LinearRegressionの予測結果の表示
y_pred1 = y_pred1.astype(np.uint32)
print(y_pred1)


# 正解の表示
print(y_test)
