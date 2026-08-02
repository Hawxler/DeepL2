# MNIST 구조 확인
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 전체 형태 확인
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape :", x_test.shape)
print("y_test shape :", y_test.shape)

print()


# 자료형 확인
print("x_train dtype:", x_train.dtype)
print("y_train dtype:", y_train.dtype)

print()


# 첫 번째 데이터 확인
print("첫 번째 이미지 shape:", x_train[0].shape)
print("첫 번째 이미지 정답:", y_train[0])

print()


# 첫 번째 이미지 픽셀 배열 일부 확인
print("첫 번째 이미지의 왼쪽 위 8x8 영역:")
print(x_train[0, :8, :8])


# 이미지 출력
plt.imshow(x_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()