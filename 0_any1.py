import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

tf.keras.backend.clear_session()
tf.keras.backend.set_image_data_format("channels_last")
tf.keras.utils.set_random_seed(42)

# --------------------------------------------------
# 1. 데이터 준비
# --------------------------------------------------
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

# 이미 RGB 채널이 있으므로 채널 차원을 추가하지 않는다.
# 입력 형태: (N, 32, 32, 3)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 정답 형태: (N, 1) → (N,)
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --------------------------------------------------
# 2. 모델 생성
# --------------------------------------------------

# 입력: 32 × 32 × 3
# 파라미터: 0개
X = layers.Input(shape=(32, 32, 3), name="input_image")

# 필터: 3 × 3, 입력 채널: 3, 필터 수: 32
# 출력 높이·너비: 32 - 3 + 1 = 30
# 출력: 30 × 30 × 32
#
# 필터 하나: 가중치 3 × 3 × 3개 + 편향 1개
# 파라미터: (3 × 3 × 3 + 1) × 32 = 896개
H = layers.Conv2D(
    32,
    kernel_size=3,
    padding="valid",
    activation="swish",
    name="conv1"
)(X)

# 필터: 3 × 3, 입력 채널: 32, 필터 수: 64
# 출력 높이·너비: 30 - 3 + 1 = 28
# 출력: 28 × 28 × 64
#
# 입력 채널은 원본 RGB의 3이 아니라
# 이전 Conv2D가 만든 특징맵의 채널 수인 32이다.
#
# 파라미터: (3 × 3 × 32 + 1) × 64 = 18,496개
H = layers.Conv2D(
    64,
    kernel_size=3,
    padding="valid",
    activation="swish",
    name="conv2"
)(H)

# 출력 높이·너비: (28 - 2) // 2 + 1 = 14
# 출력: 14 × 14 × 64
#
# 64개 특징맵 각각에 독립적으로 풀링한다.
# 채널을 서로 합치지 않으므로 채널 수는 64로 유지된다.
#
# 원소 수: 28 × 28 × 64 = 50,176개
#       → 14 × 14 × 64 = 12,544개
#
# 파라미터: 0개
H = layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=2,
    padding="valid",
    name="pool1"
)(H)

# 14 × 14 × 64 → 12,544
# 파라미터: 0개
H = layers.Flatten(name="flatten")(H)

# 입력 12,544개 → 뉴런 128개
# 파라미터: (12,544 + 1) × 128 = 1,605,760개
#
# 만약 위의 풀링층이 없다면:
# Flatten 출력: 28 × 28 × 64 = 50,176
# Dense 파라미터: (50,176 + 1) × 128 = 6,422,656개
#
# 풀링 덕분에 이 Dense 층에서
# 6,422,656 - 1,605,760 = 4,816,896개의 파라미터가 줄었다.
H = layers.Dense(
    128, activation="swish", name="dense128"
)(H)

# 입력 128개 → 뉴런 10개
# 파라미터: (128 + 1) × 10 = 1,290개
Y = layers.Dense(
    10, activation="softmax", name="output"
)(H)

model = Model(X, Y, name="cifar10_cnn_with_pooling")

# 전체 파라미터:
# 896 + 18,496 + 1,605,760 + 1,290 = 1,626,442개
model.summary()
print("전체 파라미터:", model.count_params())
exit()

# --------------------------------------------------
# 3. 학습 설정 및 학습
# --------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

# --------------------------------------------------
# 4. 평가 및 예측
# --------------------------------------------------
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"테스트 손실: {loss:.4f}")
print(f"테스트 정확도: {accuracy:.2%}")

probabilities = model.predict(x_test[:10], verbose=0)
predicted_labels = np.argmax(probabilities, axis=1)

for i in range(10):
    true_name = class_names[int(y_test[i])]
    predicted_name = class_names[int(predicted_labels[i])]

    print(
        f"{i:2d}번 이미지 | "
        f"정답: {true_name:10s} | "
        f"예측: {predicted_name:10s}"
    )