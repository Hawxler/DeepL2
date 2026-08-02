#MNIST (60000, 28, 28), CIFAR-10(50000, 32, 32, 3)
#1. MNIST 이미지 한 장 -> 내가 만든 필터 적용 -> 특징맵 출력
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

####################
#1. MNIST 데이터 셋 불러오기
####################
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, y_train: 학습용 이미지, 정답 라벨
# x_test, y_test: 테스트용 이미지, 정답 라벨

# print(x_train.shape)  # (60000, 28, 28)
# print(y_train.shape)  # (60000,)

# 첫 번째 이미지 선택
# 이미지 배열을 float32로 바꾸고 0~1 사이값으로 정규화
image = x_train[0].astype(np.float32) / 255.0
label = y_train[0]

print('이미지 크기:', image.shape)
print('정답:', label)

####################
#2. 직접 만든 필터
####################
# 세로 경계 필터
vertical_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

# 가로 경계 필터
horizontal_filter = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=np.float32)

# 평균 블러 필터
# (주변 셀과 평균적으로 너무 튀지 않고 무난하게 만듦)
blur_filter = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.float32) / 9.0 # 주변 9개 평균

# 선명하게 만드는 필터 (가운데만 뚜렷)
sharpen_filter = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

####################
#3. 단일 채널 합성곱 함수(내가 만든 합성곱)
####################
def conv2d_same(image, kernel):
    """
    2차원 흑백 이미지에 2차원 필터 하나 적용
    image   : (높이, 너비)
    kernel  : (필터 높이, 필터 너비)
    return  : 입력 이미지와 동일한 크기의 특징맵
    """

    if image.ndim != 2:
        raise ValueError("image는 2차원 흑백 이미지여야함.")
    
    if kernel.ndim != 2:
        raise ValueError("kernel은 2차원 필터여야 함.")

    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # 3x3 필터라면 상하좌우에 1칸씩 패딩
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    padded_image = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant"
    )

    output = np.zeros(
        (image_h, image_w),
        dtype=np.float32
    )

    for i in range(image_h):
        for j in range(image_w):

            # 필터 크기만큼 이미지 영역 추출
            region = padded_image[
                i : i+kernel_h,
                j : j+kernel_w
            ]

            # 원소끼리 곱한 뒤 모두 더함
            output[i, j] = np.sum(region * kernel)

    return output

####################
#4. 필터 적용
####################
vertical_map = conv2d_same(image, vertical_filter)
horizontal_map = conv2d_same(image, horizontal_filter)
blur_map = conv2d_same(image, blur_filter)
sharpen_map = conv2d_same(image, sharpen_filter)

print("원본 크기:", image.shape)
print("세로 특징맵 크기:", vertical_map.shape)
print("가로 특징맵 크기:", horizontal_map.shape)

####################
#5. 결과 시각화
####################
plt.figure(figsize=(15, 3))

plt.subplot(1, 5, 1)  # 1행 5열 격자에서 1번에 그리겠음.
plt.imshow(image, cmap='gray')
plt.title(f"Original: {label}")
plt.axis("off")

plt.subplot(1, 5, 2)  # 1행 5열 격자에서 2번에 그리겠음.
plt.imshow(np.abs(vertical_map), cmap="gray")
plt.title("Vertical edge")
plt.axis("off")

# ReLU 적용 결과: abs 결과와 비교용
# plt.subplot(1, 5, 3)
# plt.imshow(np.maximum(vertical_map, 0), cmap='gray')
# plt.title("ReLU")
# plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(np.abs(horizontal_map), cmap="gray")
plt.title("Horizontal Map")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(blur_map, cmap="gray")
plt.title("Blur")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(sharpen_map, cmap="gray")
plt.title("Sharpen")
plt.axis("off")

plt.tight_layout()
plt.show()