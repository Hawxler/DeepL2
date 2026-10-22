# Keras Functional API 코드
# 데이터 파일 다운로드(CoLab 이용 시)
# !wget https://raw.githubusercontent.com/hawklux/DeepLearn/main/1_Data/lemonade.csv

import tensorflow as tf
import pandas as pd
import numpy as np

#1. 데이터 준비
# PC 이용 시
url = "https://raw.githubusercontent.com/hawklux/DeepLearn/main/1_Data/lemonade.csv"
lemonade = pd.read_csv(url)
# lemonade = pd.read_csv('/content/lemonade.csv')
print(lemonade.head())

# X_features: X, Features, y_target: y, Target
X_features = lemonade[['온도']]   # Pandas "온도" column
y_target = lemonade[['판매량']] # Pandas "판매량" column

print(X_features.shape, y_target.shape)

#2. 모델의 구조를 만들기
# 입력층 생성
X = tf.keras.layers.Input(shape=[1])
# 완전연결층(Dense Layer) 정의
y = tf.keras.layers.Dense(1)(X)
# 전체 신경망 모델 객체 완성
model = tf.keras.models.Model(X, y)
# 학습 준비(컴파일) 완료: 학습 최적화 알고리즘(Adam), 손실함수 설정
model.compile(optimizer='adam', loss='mse')
# optim1 = tf.keras.optimizers.Adam(learning_rate=0.001,)
# model.compile(optimizer=optim1, loss='mse')

#3. 모델 학습(입력 데이터를 텐서로 변환해서 학습함)
# verbos: 0=출력 안 함, 1=상세 출력, 2=결과만 출력
model.fit(X_features, y_target, epochs=1000, verbose=0)

#4. 모델 이용 예측 (배치=1, 특성=1 형태: 15도일 때 예측 판매량)
x_new = np.array([[15.0]], dtype=np.float32)
pred = model.predict(x_new, verbose=0).ravel()[0]
print("예상 판매량:", pred)