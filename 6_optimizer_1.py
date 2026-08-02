import tensorflow as tf
import numpy as np

#1. 학습 데이터 준비 (관계: y = 2x + 1)
x_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
y_train = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]], dtype=np.float32)

#2. 아주 간단한 모델 만들기
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)  # 출력 1개
])

#3. 손실 함수와 옵티마이저 설정
loss_object = tf.keras.losses.MeanSquaredError()
learning_rate = 0.01
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

#4. 학습 1회 수행 함수
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)  # 예측값
        loss = loss_object(y, predictions)     # 손실 계산

    gradients = tape.gradient(loss, model.trainable_variables) #기울기 계산
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # 가중치 업데이트
    return loss

#5. 여러번 반복 학습
epochs = 300

for epoch in range(epochs):
    loss = train_step(x_train, y_train)

    if (epoch + 1) % 30 == 0:
        print(f"epoch={epoch+1:3d}, loss={loss.numpy():.6f}")

#6. 학습 결과 확인
w, b = model.layers[0].get_weights()
print("\n학습된 가중치 w:", w)
print("학습된 편향 b:", b)

#7. 예측 테스트
x_test = np.array([[10.0]], dtype=np.float32)
pred = model(x_test, training=False)
print("\n입력 10에 대한 예측값:", pred.numpy())