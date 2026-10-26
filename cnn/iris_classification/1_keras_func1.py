# Keras functional API 모델
# Data preprocess > model > run

######################################
# 1. 데이터 준비(0_dataPrep1.py의 내용)
######################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1.1 데이터 읽기
url = "https://raw.githubusercontent.com/hawklux/DeepLearn/refs/heads/main/0_Data/bong/iris.csv"
data = pd.read_csv(url)

print(data.shape)
print(data.head())
print(data.columns)

# 1.2 입력 특성
feature_columns = ['꽃받침길이', '꽃받침폭', '꽃잎길이', '꽃잎폭']

X = data[feature_columns].to_numpy(dtype=np.float32)

# 1.3 정답의 클래스 순서를 고정
class_names = np.array(['setosa', 'versicolor', 'virginica'])
class_to_index = {
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2,
}

y_index = data['품종'].map(class_to_index)
# print(y_index) # 이미지번호 + 품종클래스 번호
if y_index.isna().any():
    raise ValueError("품종 값 오류")

# 1.4 데이터 shuffle + 분할: 80% 학습, 20% 테스트
X_train_raw, X_test_raw, y_train_index, y_test_index = train_test_split(
    X,
    y_index,
    test_size=0.2,
    random_state=42,
    stratify=y_index,
)

# 1.5 입력 표준화: 학습 데이터로만 평균과 표준편차 계산
# fit_transform(훈련 데이터): fit(평균, 표준편차 구하기) + transform(변환)
# transform(테스트 데이터): 앞서 구해둔 동일한 기준(평균, 표준편차) 적용해서 변환
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test = scaler.transform(X_test_raw).astype(np.float32)
print(f"학습용 입력: {X_train.shape}, {X_train[:1]}" )
print(f"테스트용 입력: {X_test.shape}, {X_test[:1]}")

# 6. 출력 원-핫 인코딩
# 0 -> [1, 0, 0], 1 -> [0, 1, 0], 2 -> [0, 0, 1]
y_train = np.eye(3, dtype=np.float32)[y_train_index]
y_test = np.eye(3, dtype=np.float32)[y_test_index]
print(f"학습용 정답: {y_train.shape}, {y_train[0]}") #(120, 3), [1. 0. 0.]
print(f"테스트 정답: {y_test.shape}, {y_test[:1]}") #(30, 3), [[1. 0. 0.]]
print("섞인 행 인덱스, 품종 인덱스:\n", y_train_index[:5])
print("One-hot 정답:\n", y_train[:5])

# 7. 추론 연습용 데이터 스케일링
# 순서: 꽃받침길이, 꽃받침폭, 꽃잎길이, 꽃잎폭
X_new_raw = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.5, 3.0, 5.8, 2.2]
], dtype=np.float32)

# Scaling 하기
# transform(): 추론도 학습 때 구한 동일한 기준(평균, 표준편차) 사용
X_new = scaler.transform(X_new_raw).astype(np.float32)
print(X_new[:2])

######################################
# 2. 모델 구축(Keras Functional API)
######################################
import tensorflow as tf
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42) 
# 코드 실행할 때마다 각 레이어의 Dense()가 동일한 랜덤값으로 최초 (필터)의 weight, bias를 넣어주고 학습을 시작하게 해줌. (단, 랜덤값이 전체 데이터에 비해 이상값이 들어가지 않게 디폴트로 glorot_uniform 알고리즘을 적용해줌. ReLu는 he_normal 알고리즘이 더 잘 맞기 때문에 Dense(3, activation='ReLU', kernel_initializer='he_normal')로 지정해줄 수 있음.) 이렇게 고정해야 추후 모델 최적화를 위해 파라미터 수정을 해도 무엇 때문에 성능의 변화가 생겼는지 파악할 수 있게 됨.

#2.1 모델 구성
#(1) 입력층: 특성 4개
inputs = tf.keras.Input(shape=(4,))

#(2) 첫 번째 은닉층: 뉴런 16개
H = layers.Dense(16)(inputs)
H = layers.BatchNormalization()(H)
H = layers.ReLU()(H)
H = layers.Dropout(0.2)(H)

#(3) 두 번째 은닉층: 뉴런 8개
# H = layers.Dense(8, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(H)
# H = layers.BatchNormalization()(H)
H = layers.Dense(8)(H)
H = layers.BatchNormalization()(H)
H = layers.ReLU()(H)

#(4) 출력층: 품종별 확률 3개
outputs = layers.Dense(3, activation='softmax')(H)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
)

model.summary()

#################################################
# 3. 학습 > 평가 > 추론: Sequential/Functinal 동일
#################################################
#3.1 학습 설정
optim1 = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optim1, 
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
)

#3.2 학습
# fit() 내부에서 반복: 순전파 > 손실 계산 > 역전파 > 가중치 갱신
history = model.fit(
    X_train,
    y_train,
    batch_size=16,
    epochs=200,
    shuffle=True,
    verbose=0,
)

history = model.fit(X_train, y_train, batch_size=16, epochs=3, shuffle=True, verbose=1)

#3.3 테스트 데이터 평가
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"테스트 손실: {test_loss:.4f}")
print(f"테스트 정확도: {test_accuracy:.2%}")

#3.4 품종 예측
probabilities = model.predict(X_new, verbose=0)

# 각 행에서 가장 큰 확률의 열 번호 선택
predicted_indices = np.argmax(probabilities, axis=1)

for i, class_index in enumerate(predicted_indices):
    print(
        f"입력: {X_new_raw[i]} | "
        f"예측 품종: {class_names[class_index]} | "
        f"확률: {np.round(probabilities[i], 4)}"
    )
