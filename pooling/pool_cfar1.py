import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10


# --------------------------------------------------
# 1. 컬러 이미지 풀링 함수
# --------------------------------------------------
def pool2d_color(image, pool_size=2, stride=2, mode="max"):
    """
    RGB 이미지에 채널별로 풀링을 적용한다.

    image 형태: (높이, 너비, 채널)
    """

    image_h, image_w, channels = image.shape

    output_h = (image_h - pool_size) // stride + 1
    output_w = (image_w - pool_size) // stride + 1

    output = np.zeros(
        (output_h, output_w, channels),
        dtype=np.float32
    )

    for i in range(output_h):
        for j in range(output_w):

            start_i = i * stride
            start_j = j * stride

            # 모양: (pool_size, pool_size, 채널)
            region = image[
                start_i:start_i + pool_size,
                start_j:start_j + pool_size,
                :
            ]

            if mode == "max":
                # 높이와 너비 방향으로 최대값 계산
                # R, G, B 채널은 유지
                output[i, j, :] = np.max(
                    region,
                    axis=(0, 1)
                )

            elif mode == "average":
                output[i, j, :] = np.mean(
                    region,
                    axis=(0, 1)
                )

            else:
                raise ValueError(
                    'mode는 "max" 또는 "average"여야 합니다.'
                )

    return output


# --------------------------------------------------
# 2. CIFAR-10 불러오기
# --------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# --------------------------------------------------
# 3. 이미지 한 장 선택
# --------------------------------------------------
image_index = 0

image = x_train[image_index].astype(np.float32) / 255.0

label_number = int(y_train[image_index, 0])
label_name = class_names[label_number]


# --------------------------------------------------
# 4. 풀링 적용
# --------------------------------------------------
max_pooled = pool2d_color(
    image,
    pool_size=2,
    stride=2,
    mode="max"
)

average_pooled = pool2d_color(
    image,
    pool_size=2,
    stride=2,
    mode="average"
)


# --------------------------------------------------
# 5. 결과 확인
# --------------------------------------------------
print("클래스 번호:", label_number)
print("클래스 이름:", label_name)

print("원본 크기:", image.shape)
print("Max Pooling 크기:", max_pooled.shape)
print("Average Pooling 크기:", average_pooled.shape)


# --------------------------------------------------
# 6. 이미지 비교
# --------------------------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title(
    f"Original: {label_name}\n{image.shape}"
)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(np.clip(max_pooled, 0, 1))
plt.title(
    f"Max Pooling\n{max_pooled.shape}"
)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(np.clip(average_pooled, 0, 1))
plt.title(
    f"Average Pooling\n{average_pooled.shape}"
)
plt.axis("off")

plt.tight_layout()
plt.show()