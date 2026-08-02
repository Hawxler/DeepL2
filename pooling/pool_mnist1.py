import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


# --------------------------------------------------
# 1. 풀링 함수
# --------------------------------------------------
def pool2d(image, pool_size=2, stride=2, mode="max"):
    """
    2차원 흑백 이미지에 풀링을 적용한다.

    image     : (높이, 너비)
    pool_size : 풀링 영역 크기
    stride    : 이동 간격
    mode      : "max" 또는 "average"
    """

    image_h, image_w = image.shape

    output_h = (image_h - pool_size) // stride + 1
    output_w = (image_w - pool_size) // stride + 1

    output = np.zeros(
        (output_h, output_w),
        dtype=np.float32
    )

    for i in range(output_h):
        for j in range(output_w):

            start_i = i * stride
            start_j = j * stride

            region = image[
                start_i:start_i + pool_size,
                start_j:start_j + pool_size
            ]

            if mode == "max":
                output[i, j] = np.max(region)

            elif mode == "average":
                output[i, j] = np.mean(region)

            else:
                raise ValueError(
                    'mode는 "max" 또는 "average"여야 합니다.'
                )

    return output


# --------------------------------------------------
# 2. MNIST 데이터 불러오기
# --------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_index = 0

image = x_train[image_index].astype(np.float32) / 255.0
label = int(y_train[image_index])


# --------------------------------------------------
# 3. 풀링 적용
# --------------------------------------------------
max_pooled = pool2d(
    image,
    pool_size=2,
    stride=2,
    mode="max"
)

average_pooled = pool2d(
    image,
    pool_size=2,
    stride=2,
    mode="average"
)


# --------------------------------------------------
# 4. 크기 확인
# --------------------------------------------------
print("정답:", label)
print("원본 크기:", image.shape)
print("Max Pooling 크기:", max_pooled.shape)
print("Average Pooling 크기:", average_pooled.shape)


# --------------------------------------------------
# 5. 이미지 비교
# --------------------------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title(f"Original: {label}\n{image.shape}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(max_pooled, cmap="gray")
plt.title(f"Max Pooling\n{max_pooled.shape}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(average_pooled, cmap="gray")
plt.title(f"Average Pooling\n{average_pooled.shape}")
plt.axis("off")

plt.tight_layout()
plt.show()