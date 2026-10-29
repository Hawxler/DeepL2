# Keras Sequential API 코드
import tensorflow as tf
import pandas as pd
import numpy as np

# 1. 데이터 준비
url = "https://raw.githubusercontent.com/hawklux/DeepLearn/refs/heads/main/0_Data/bong/lemonade.csv"
data = pd.read_csv(url)
print(data.head())

# inVar: 온도, tarVar: 판매량
# float32 Numpy 배열로 변환
inVar = data[['온도']].to_numpy(dtype=np.float32)
tarVar = data[['판매량']].to_numpy(dtype=np.float32)

print("입력(샘플 수, 특성 수):", inVar.shape)
print("출력(행 수, 열 수):", tarVar.shape)

# 2. 모델 구조: Sequential 방식
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)), #샘플당 입력 특성 1개
    tf.keras.layers.Dense(1),   #샘플당 출력값 1개
])

# 3. 학습 설정
optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer1, loss='mse')
model.summary()

# 4. 모델 학습 (batch: 32, 64, 128 해도 됨. 에폭마다 shuffle해줘서 암기 방지)
model.fit(inVar, tarVar, batch_size=32, epochs=1000, shuffle=True, verbose=0)

# 5. 모델 사용: 예측
# 샘플 1개, 특성 1개 -> shape: (1, 1)
X_new = np.array([[15.0]], dtype=np.float32)
pred = model.predict(X_new, verbose=1).ravel()[0]
print(f"15도 일 때 예상 판매량: {pred:.2f}")

# 6. 학습한 가중치와 편향 확인
# Input()은 model.layers()가 아님
# 즉, model.layers[0]은 Dense 층임.
w, b = model.layers[0].get_weights()

print(f"가중치 w: {w[0, 0]:.4f}")
print(f"편향 b: {b[0]:.4f}")