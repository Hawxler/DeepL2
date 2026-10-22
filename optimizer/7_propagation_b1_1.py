# 7_propagation_f1.py에 대응하는 역전파 코드
# Keras + Tensorflow 

import numpy as np
import tensorflow as tf
from tensorflow import keras

# --------------------------------------------------
# 1. 입력과 정답
# 입력 형태: (데이터 개수, 입력 특성 개수)
# --------------------------------------------------
x = tf.constant([[1.0, 2.0]], dtype=tf.float64)
y = tf.constant([[1.0]], dtype=tf.float64)

# --------------------------------------------------
# 2. 모델: 입력 2개 → 은닉 뉴런 2개 → 출력 1개
# 그림과 동일하게 편향은 사용하지 않음
# --------------------------------------------------
model = keras.Sequential([
    keras.Input(shape=(2,), dtype="float64"),

    keras.layers.Dense(
        2,
        activation="sigmoid",
        use_bias=False,
        dtype="float64",
        name="hidden",
    ),

    keras.layers.Dense(
        1,
        activation="sigmoid",
        use_bias=False,
        dtype="float64",
        name="output",
    ),
])

# Keras의 가중치 배열: (입력 수, 출력 뉴런 수)
#
# 은닉층:
# [[w1, w2],
#  [w3, w4]]
#
# 출력층:
# [[w5],
#  [w6]]
model.set_weights([
    np.array([
        [0.3, 0.2],
        [0.2, 0.3],
    ], dtype=np.float64),

    np.array([
        [0.4],
        [0.5],
    ], dtype=np.float64),
])

# 데이터 1개, 출력 1개이므로 MSE = (y - s3)^2
loss_fn = keras.losses.MeanSquaredError()

# 이전 수동 계산과 동일한 경사하강법
optimizer = keras.optimizers.SGD(learning_rate=0.1)


def print_weights():
    hidden, output = model.get_weights()

    values = [
        hidden[0, 0],  # w1
        hidden[0, 1],  # w2
        hidden[1, 0],  # w3
        hidden[1, 1],  # w4
        output[0, 0],  # w5
        output[1, 0],  # w6
    ]

    for i, value in enumerate(values, start=1):
        print(f"w{i} = {value:.6f}")


# --------------------------------------------------
# 3. 순전파와 비용 계산
# GradientTape가 계산 과정을 기록
# --------------------------------------------------
with tf.GradientTape() as tape:
    pred_before = model(x, training=True)
    loss_before = loss_fn(y, pred_before)

# --------------------------------------------------
# 4. 역전파: 모든 가중치의 기울기 계산
# 이 줄에서는 가중치가 바뀌지 않음
# --------------------------------------------------
gradients = tape.gradient(
    loss_before,
    model.trainable_variables,
)

# 가중치 배열과 같은 위치에 기울기가 들어 있음
gh = gradients[0].numpy()
go = gradients[1].numpy()

gradient_values = [
    gh[0, 0], gh[0, 1],
    gh[1, 0], gh[1, 1],
    go[0, 0], go[1, 0],
]

print("=== 역전파로 구한 기울기 ===")
for i, value in enumerate(gradient_values, start=1):
    print(f"dC/dw{i} = {value:.6f}")

# --------------------------------------------------
# 5. 옵티마이저가 가중치 갱신
# w_new = w_old - learning_rate * gradient
# --------------------------------------------------
optimizer.apply_gradients(
    zip(gradients, model.trainable_variables)
)

print("\n=== 갱신된 가중치 ===")
print_weights()

# --------------------------------------------------
# 6. 갱신된 가중치로 다시 순전파 및 비용 계산
# --------------------------------------------------
pred_after = model(x, training=False)
loss_after = loss_fn(y, pred_after)

print("\n=== 학습 전후 비교 ===")
print(
    f"예측값: {pred_before.numpy().item():.6f}"
    f" → {pred_after.numpy().item():.6f}"
)
print(
    f"비용:   {loss_before.numpy().item():.6f}"
    f" → {loss_after.numpy().item():.6f}"
)