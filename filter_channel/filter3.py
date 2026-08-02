# 3차원 CIFAR-10 이미지에 필터 적용 예
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D

#############################################
#1. CIFAR-10 불러오기
#############################################
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train, y_train: 학습용 이미지, 정답 라벨
# x_test, y_test: 테스트용 이미지, 정답 라벨

# print(x_train.shape, y_train.shape) #(50000,32,32,3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000,32,32,3) (10000, 1)
# print(x_train[0].shape, y_train[0]) #첫째 이미지(32, 32, 3) [6]
# exit()

class_names = ["airplane", "automobile", "bird", "cat", 
               "deer", "dog", "frog", "horse", "ship", "truck"]

# 첫번째 이미지 선택
# 이미지 배열을 float32로 바꾸고 0~1 사이값으로 정규화
image = x_train[0].astype(np.float32) / 255.0
label = int(y_train[0, 0])

print("입력 이미지 크기:", image.shape)
print("정답:", class_names[label])

# Conv2D는 batch 차원까지 필요함. (batch, h, w, 채널)
# (32, 32, 3) 이미지 1장 -> (1, 32, 32, 3)처럼 np.newaxis로
# 맨 앞에 새 차원을 추가하여 "이미지 한 장 짜리 기본 배치"로 만듦.
# 맨 뒤 추가: [:, np.newaxis]
# ... = :, :, : = 높이 전체, 너비 전체, 채널 전체
image_batch = image[np.newaxis, ...]

print("배치 포함 크기:", image_batch.shape)

#############################################
#2. 세로 경계 기본 필터
#############################################
vertical_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

#############################################
#3. Conv2D 계층 만들기: Keras의 합성곱 수행
#############################################
conv_layer = Conv2D(
    filters=2,          # 필터 2개
    kernel_size=3,      # 3x3
    padding='same',
    use_bias=False,
    activation=None,
    trainable=False
)

# 계층의 입력/가중치 모양을 만들기 위해 한 번 실행
# Conv2D 계층은 처음 만들어질 때 아직 입력 채널이 몇 개인지 모름.
# 그래서, image_batch를 넣고 한 번 돌려줌. 그러면
# image_batch.shape인 (1,32,32,3)의 3을 받아 채널을 알게됨. 
_ = conv_layer(image_batch)

#############################################
#4. 직접 가중치 배열 만들기
#############################################
'''
모양: (필터 높이, 필터 너비, 입력 채널, 필터 개수)
즉, (3, 3, 3, 2)
'''
weights = np.zeros(
    (3, 3, 3, 2),
    dtype=np.float32
)

# 필터 0:
# R, G, B 세 채널에 같은 세로 경계 필터 적용
# 세 채널을 더하므로 값이 너무 커지지 않도록 3으로 나눔
weights[:, :, 0, 0] = vertical_filter / 3.0  # R
weights[:, :, 1, 0] = vertical_filter / 3.0  # G
weights[:, :, 2, 0] = vertical_filter / 3.0  # B

# 필터 1:
# R 채널에만 세로 경계 필터 적용
weights[:, :, 0, 1] = vertical_filter   # R
weights[:, :, 1, 1] = 0                 # G
weights[:, :, 2, 1] = 0                 # B

# 직접 만든 필터를 Conv2D 계층에 입력
conv_layer.set_weights([weights])

#############################################
#5. 필터 적용
#############################################
feature_maps = conv_layer(image_batch).numpy()

print("필터 가중치 크기:", conv_layer.get_weights()[0].shape)
print("출력 전체 크기:", feature_maps.shape)

# Batch 차원 제거
# (1, 32,32, 1) -> (32, 32, 2)
feature_maps = feature_maps[0]

all_channel_edge = feature_maps[:, :, 0]
red_channel_edge = feature_maps[:, :, 1]

#############################################
#6. 결과 출력
#############################################
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title(f"Original: {class_names[label]}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(np.abs(all_channel_edge), cmap="gray")
plt.title("RGB verified edge")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(np.abs(red_channel_edge), cmap="gray")
plt.title("Red channel edge")
plt.axis("off")

plt.tight_layout()
plt.show()