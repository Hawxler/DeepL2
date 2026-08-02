''' 연습 순서
#1. 1채널 입력 + 필터 1개
#2. 1채널 입력 + 필터 여러 개
#3. 다채널 입력 + 다채널 필터
#4. 풀링
'''
#1. 1채널 입력 + 필터 1개
# 입력 이미지: 5x5x1
# 필터: 3x3x1
# 출력: 3x3x1 특징맵

import numpy as np

# 5x5 흑백 이미지
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=float)

# 3x3 세로 경계 필터
kernel1 = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=float)

def conv2d(image, kernel):
    # 2차원 입력에 2차원 필터 하나 적용
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # 출력 특징맵 크기: 패딩 없음, 스트라이드 1
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1

    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):

            # 입력 이미지에서 필터 크기만큼 자른 영역
            region = image[
                i:i + kernel_h,
                j:j + kernel_w
            ]

            # 같은 위치끼리 곱한 후 모두 더한다.
            output[i, j] = np.sum(region * kernel)

    return output

feature_map = conv2d(image, kernel1)

print("입력 이미지 크기:", image.shape)
print("필터 크기:", kernel1.shape)
print("특징맵 크기:", feature_map.shape)

print("\n특징맵:")
print(feature_map)