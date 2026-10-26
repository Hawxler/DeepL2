# Keras Sequential API 코드

######################################
# 1. 데이터 준비(0_dataPrep1.py의 내용)
######################################
# 데이터 준비: 불러오기 > 입력/정답 분리 > train_test_split > 입력 표준화(StandardScaler) > 정답 One-hot encoding
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 읽기
url = "https://raw.githubusercontent.com/hawklux/DeepLearn/refs/heads/main/0_Data/bong/iris.csv"
data = pd.read_csv(url)

print(data.shape)
print(data.head())
print(data.columns)

# 2. 입력 특성
feature_columns = ['꽃받침길이', '꽃받침폭', '꽃잎길이', '꽃잎폭']

X = data[feature_columns].to_numpy(dtype=np.float32)

# 3. 정답의 클래스 순서를 고정
class_names = np.array(['setosa', 'versicolor', 'virginica'])
class_to_index = {
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2,
}

y_index = data['품종'].map(class_to_index)
if y_index.isna().any():
    raise ValueError("품종 값 오류")

# 4. 데이터 Shuffle + 분할: 80% 학습, 20% 테스트
X_train_raw, X_test_raw, y_train_index, y_test_index = train_test_split(
    X,
    y_index,
    test_size=0.2,
    random_state=42,
    stratify=y_index,
)

# 5. 표준화: 학습 데이터로만 평균과 표준편차 계산
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test = scaler.transform(X_test_raw).astype(np.float32)
print("학습용 입력:", X_train.shape, X_train[:1])
print("테스트용 입력:", X_test.shape, X_test[:1])

# 6. 원-핫 인코딩
# 0 -> [1, 0, 0], 1 -> [0, 1, 0], 2 -> [0, 0, 1]
y_train = np.eye(3, dtype=np.float32)[y_train_index]
y_test = np.eye(3, dtype=np.float32)[y_test_index]

print("테스트 입력:", y_train.shape, y_train[0])  # (120, 3) [1. 0. 0.]
print("테스트 정답:", y_test.shape, y_test[:1])   # (30, 3) [[1. 0. 0.]]
print("섞인 행 인덱스, 품종 인덱스:\n", y_train_index[:5])
print("One-hot 정답:\n", y_train[:5])

# 7. 추론 연습용 데이터
# 순서: 꽃받침길이, 꽃받침폭, 꽃잎길이, 꽃잎폭
X_new_raw = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.5, 3.0, 5.8, 2.2]
], dtype=np.float32)

# 추론도 학습 때 구한 동일한 기준 사용
# transform(): 테스트/새 입력 시 테스트 데이터로 기준을 다시 학습하지 않음.
X_new = scaler.transform(X_new_raw).astype(np.float32)
print(X_new[:2])

######################################
# 2. 모델 구축(Keras Sequential API)
######################################
import tensorflow as tf
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)

model = tf.keras.Sequential([
    # 입력층
    tf.keras.Input(shape=(4,)),

    # 은닉층1: 뉴런 16개
    layers.Dense(16),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.2),

    # 은닉층2: 뉴런 8개
    layers.Dense(8),
    layers.BatchNormalization(),
    layers.ReLU(),

    # 출력층: 픔종별 확률 3개
    layers.Dense(3, activation='softmax')
])

model.summary()

#################################################
# 3. 학습 > 평가 > 추론: Sequential/Functinal 동일
#################################################
# 3.1 학습 설정
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # 혹은 아래처럼 상술해도 좋음
# optim1 = tf.keras.optimizers.Adam(learning_rate=0.001)
# model.compile(
#     optimizer=optim1,
#     loss=tf.keras.losses.CategoricalCrossentropy(),
#     metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
# )

# 3.2 학습
# fit() 내부 반복: 순전파 > 손실 계산 > 역전파 > 가중치 갱신
history = model.fit(
    X_train,
    y_train,
    batch_size=16,
    epochs=200,
    shuffle=True,
    verbose=0,
)

history = model.fit(X_train, y_train, batch_size=16, epochs=3, shuffle=True, verbose=2)

# 3.3 테스트 데이터 평가
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
print(f"테스트 손실: {test_loss:.4f}")
print(f"테스트 정확도: {test_accuracy:.2%}")

# 3.4 품종 예측
probabilities = model.predict(X_new, verbose=0)

# 각 행에서 가장 큰 확률의 열 번호 선택
predicted_indices = np.argmax(probabilities, axis=1)
for i, class_index in enumerate(predicted_indices):
    print(
        f"입력: {X_new_raw[i]} | "
        f"예측 품종: {class_names[class_index]} | " 
        f"확률: {np.round(probabilities[i], 4)}"
    )