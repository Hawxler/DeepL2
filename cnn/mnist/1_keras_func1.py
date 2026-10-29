# Keras Functional API 코드

#########################################
#1. 이미지 전처리 (0_DataPrep1.py 내용)
# MNIST 데이터셋 전처리
# 28 x 28 흑백, 학습용: 60,000, 테스트용 10,000, 검증용 6,000
# 현실 이미지 경우 예: 648x480 -> 패딩추가-> 648x648 -> 224x224
#########################################

# 원-핫 없이 정수 정답 0~9 사용해보기
import os
from urllib.request import urlretrieve

import numpy as np
from sklearn.model_selection import train_test_split

SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE = 42, 5, 64, 0.001

# 1.1 데이터 다운로드 (현재 폴더 아래 data/mnist.npz)
try:
    # 일반 파이썬(.py) 파일로 실행할 때
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 주피터 노트북(Jupyter Notebook)이나 구글 코랩(Colab)에서 실행할 때는 __file__ 파일 경로가 없으로 아래처럼.
    script_dir = os.getcwd()

data_dir = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True) # data 폴더가 없으면 생성

data_path = os.path.join(data_dir, "mnist.npz")

if not os.path.exists(data_path): # data_path에 mnist.npz가 없다면
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    print("MNIST 다운 중...")
    urlretrieve(url, data_path)
    print("다운로드 완료!")

# 1.2 데이터 읽기(npz -> all + test)
with np.load(data_path, allow_pickle=False) as data: # 보안: 피클은 차단
    X_all_raw = data["x_train"]
    y_all = data['y_train']

    X_test_raw = data["x_test"]
    y_test = data["y_test"]

# 1.3 데이터 살펴보기 (전체 출력 대신 안전하게 shape과 샘플만 출력)
np.set_printoptions(linewidth=200)
print("X_all_raw shape:", X_all_raw.shape)  # (60000, 28, 28)
print("첫 번째 이미지 픽셀:\n", X_all_raw[:1]) # 5 이미지 픽셀
print("첫 번째 이미지의 9번째 픽셀값:", X_all_raw[0][8])
print("첫 번째 이미지 shape:", X_all_raw[0].shape) # (28, 28)
print("첫 번째 정답(y):", y_all[0])                # 5
print("y_all shape:", y_all.shape)                # (60000,)

# 1.4 학습 데이터에서 검증 데이터 10% 분리 (all -> train + val)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_all_raw,
    y_all,
    test_size=0.1,
    random_state=SEED,
    stratify=y_all, # 0~9 이미지 누락 없이 골고루 넣기
)

# 1.5 픽셀 값 정규화: 0~255 -> 0~1
def preprocess_images(images):
    return np.asarray(images, dtype=np.float32) / 255.0

X_train = preprocess_images(X_train_raw)
X_val = preprocess_images(X_val_raw)
X_test = preprocess_images(X_test_raw)

# 1.6 정답: 정수 클래스 번호
y_train = y_train.astype(np.int64)
y_val = y_val.astype(np.int64)
y_test = y_test.astype(np.int64)

print("학습:", X_train.shape, y_train.shape) # (54000, 28, 28) (54000,)
print("검증:", X_val.shape, y_val.shape)     # (6000, 28, 28) (6000,)
print("테스트:", X_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

######################################
# 2. 모델 구축(Keras Functional API)
######################################
import tensorflow as tf
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(SEED)

# 1. 입력층
inputs = tf.keras.Input(shape=(28, 28, 1)) # 1채널 회색조

# 합성곱층 1
H = layers.Conv2D(
    filters=16, kernel_size=3, padding="valid",
    use_bias=False, kernel_initializer="he_normal",
)(inputs)  # 28 - (3 -1): 26x26 16개 특성맵 생성
H = layers.BatchNormalization()(H)
H = layers.ReLU()(H)
# 최대 풀링층
H = layers.MaxPooling2D(pool_size=2)(H)  # 13x13 16개 특성맵

# 합성곱층 2
H = layers.Conv2D(
    filters=32, kernel_size=3, padding="same",
    use_bias=False, kernel_initializer="he_normal",
)(H)  # padding "same" 13: 13x13 32개 특성맵 생성
H = layers.BatchNormalization()(H)
H = layers.ReLU()(H)
# 최대 풀링층
H = layers.MaxPooling2D(pool_size=2)(H) # 6x6 32개 특성맵

# 완전연결층: 분류 부분
H = layers.Flatten()(H)
H = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(H)
H = layers.Dropout(0.2)(H)
outputs = layers.Dense(10, activation="softmax")(H)

model = tf.keras.Model(inputs, outputs)
model.summary()

######################################
# 3. 학습
######################################
# 3.1 채널 축 추가: 전처리 데이터 3D + 채널 = CNN 레이어 4D
# X_train(54000, 28, 28) -> CNN layer(배치크기, 세로, 가로, 채널) -> (54000, 28, 28, 1)
# ...: 나머지는 기존대로 유지
X_train_keras = X_train[..., np.newaxis]
X_val_keras = X_val[..., np.newaxis]
X_test_keras = X_test[..., np.newaxis]

# 3.2 학습 설정: softmax 출력 + 정수 정답
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

# 3.3 학습
history = model.fit(
    X_train_keras,
    y_train,
    validation_data=(X_val_keras, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=2,
)

######################################
# 4. 평가/추론
######################################
# 4.1 테스트 데이터로 최종 평가
test_loss, test_accuracy = model.evaluate(
    X_test_keras, y_test, batch_size=BATCH_SIZE, verbose=0
)
print(f"테스트 손실: {test_loss:.4f}")
print(f"테스트 정확도: {test_accuracy:.2%}")

# 4.2 추론 함수
# 입력: 정규화 전 이미지, shape=(N, 28, 28), 값 범위 0~255
def predict_keras(model, raw_images):
    images = preprocess_images(raw_images)
    images = images[..., np.newaxis]

    probabilities = model.predict(images, verbose=0)
    predictions = probabilities.argmax(axis=1)
    return predictions, probabilities

# 4.3 테스트 이미지 5장으로 추론 사용법 확인
predictions, probabilities = predict_keras(model, X_test_raw[:5])

for i, predicted in enumerate(predictions):
    confidence = probabilities[i, predicted]
    print(
        f'이미지 {i}: 정답={y_test[i]}, '
        f"예측={predicted}, 확률={confidence:.2%}"
    )