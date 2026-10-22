import numpy as np
import tensorflow as tf

keras = tf.keras

# shape=(1, 4): 데이터 1개, 특성 4개
x = np.array([[2.0, 4.0, 6.0, 8.0]], dtype=np.float32)

inputs = keras.Input(shape=(4,))

# 입력을 그대로 전달하도록 초기 가중치 설정(eye: 대각선 1 행렬)
hidden = keras.layers.Dense(
    units=4,
    use_bias=False,
    kernel_initializer=keras.initializers.Constant(
        np.eye(4, dtype=np.float32)
    ),
)(inputs)

dropped = keras.layers.Dropout(rate=0.5)(hidden)

# 네 값을 더하도록 초기 가중치 설정
output = keras.layers.Dense(
    units=1,
    use_bias=False,
    kernel_initializer="ones",
)(dropped)

# 중간 결과까지 관찰하기 위한 모델
model = keras.Model(
    inputs=inputs,
    outputs=[hidden, dropped, output],
)

h, d, y = model(x, training=True)

print("학습 모드")
print("Dense 출력:", h.numpy())
print("Dropout 출력:", d.numpy())
print("최종 출력:", y.numpy())

h, d, y = model(x, training=False)

print("\n추론 모드")
print("Dense 출력:", h.numpy())
print("Dropout 출력:", d.numpy())
print("최종 출력:", y.numpy())