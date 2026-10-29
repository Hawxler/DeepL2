import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. 데이터 준비 (MNIST 데이터셋 및 32x32 패딩 처리)
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# LeNet-5 표준 입력 크기(32x32)에 맞추기 위해 패딩 추가 (28x28 -> 32x32)
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')

# 정규화 (0~255 -> 0~1) 및 채널 차원 추가 (N, 32, 32, 1)
X_train = X_train[..., np.newaxis].astype("float32") / 255.0
X_test = X_test[..., np.newaxis].astype("float32") / 255.0

# 2. LeNet-5 모델 구축 (Functional API)
inputs = tf.keras.Input(shape=(32, 32, 1))  # 32x32 이미지 한 장
H = layers.Conv2D(6, kernel_size=5, activation='tanh', padding='valid')(inputs) # 28x28 특성맵 6장 만들어짐
H = layers.AveragePooling2D(pool_size=2, strides=2)(H)  
H = layers.Conv2D(16, kernel_size=5, activation='tanh', padding='valid')(H)
H = layers.AveragePooling2D(pool_size=2, strides=2)(H)
H = layers.Flatten()(H)
H = layers.Dense(120, activation='tanh')(H)
H = layers.Dense(84, activation='tanh')(H)
outputs = layers.Dense(10, activation='softmax')(H)

model = tf.keras.Model(inputs, outputs, name="LeNet-5")
model.summary()

# 3. 모델 컴파일 및 학습
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("모델 학습 시작...")
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)

# 4. 테스트 데이터 평가
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 손실: {test_loss:.4f}, 테스트 정확도: {test_acc:.2%}")

# 5. 추론(예측) 예시
predictions = model.predict(X_test[:5], verbose=0)
predicted_classes = predictions.argmax(axis=1)
print("예측 결과:", predicted_classes)
print("실제 정답:", y_test[:5])