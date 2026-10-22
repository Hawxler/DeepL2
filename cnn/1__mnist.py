import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

print()

print("x_train dtype:", x_train.dtype)
print("y_train dtype:", y_train.dtype)

print()

print("첫 번째 이미지 shape:", x_train[0].shape)
np.set_printoptions(linewidth=200)
print("첫 번째 이미지 정답:", y_train[0])

print()

print(x_train[0])

print(x_train[0, :8, :8])


plt.imshow(x_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()