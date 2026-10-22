import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10


# --------------------------------------------------
# 1. CIFAR-10 데이터 불러오기
# --------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# --------------------------------------------------
# 2. CIFAR-10 클래스 이름
# --------------------------------------------------
# 인덱스가 곧 CIFAR-10의 라벨 번호이다.
class_names = [
    "airplane",    # 0: 비행기
    "automobile",  # 1: 자동차
    "bird",        # 2: 새
    "cat",         # 3: 고양이
    "deer",        # 4: 사슴
    "dog",         # 5: 개
    "frog",        # 6: 개구리
    "horse",       # 7: 말
    "ship",        # 8: 배
    "truck"        # 9: 트럭
]


# --------------------------------------------------
# 3. 전체 데이터 형태 확인
# --------------------------------------------------
print("===== 전체 데이터 형태 =====")
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape :", x_test.shape)
print("y_test.shape :", y_test.shape)

print()

print("===== 자료형 =====")
print("x_train.dtype:", x_train.dtype)
print("y_train.dtype:", y_train.dtype)

print()

print("===== 픽셀값 범위 =====")
print("최솟값:", x_train.min())
print("최댓값:", x_train.max())


# --------------------------------------------------
# 4. 클래스 번호와 이름 출력
# --------------------------------------------------
print("\n===== 클래스 번호와 이름 =====")

for class_number, class_name in enumerate(class_names):
    print(f"{class_number}: {class_name}")


# --------------------------------------------------
# 5. 첫 번째 학습 이미지 확인
# --------------------------------------------------
image_index = 0

image = x_train[image_index]
# print(image)
print(len(image[:1][0]))
print(image[:1][0])

# y_train은 [[3] [8] [8] [0] [6]...] 같은 2차원 배열
# y_train[0][0] -> 3
print("y_train[:4]:", y_train[:4])  #[[3] [8] [8] [0]]
label_number = int(y_train[image_index, 0])
label_name = class_names[label_number]

print("\n===== 첫 번째 이미지 =====")
print("이미지 인덱스:", image_index)
print("이미지 shape:", image.shape)
print("라벨 원본:", y_train[image_index])
print("라벨 번호:", label_number)
print("클래스 이름:", label_name)


# --------------------------------------------------
# 6. 특정 픽셀의 RGB 값 확인
# --------------------------------------------------
row = 10
column = 20

pixel = image[row, column]

print("\n===== 특정 픽셀 확인 =====")
print(f"위치: {row}행 {column}열")
print("RGB 배열:", pixel)
print("R 값:", pixel[0])
print("G 값:", pixel[1])
print("B 값:", pixel[2])


# --------------------------------------------------
# 7. 첫 번째 이미지 출력
# --------------------------------------------------
plt.figure(figsize=(4, 4))

plt.imshow(image)
plt.title(f"Label {label_number}: {label_name}")
plt.axis("off")

plt.show()


# --------------------------------------------------
# 8. 앞의 10개 이미지 출력
# --------------------------------------------------
plt.figure(figsize=(15, 6))

for i in range(10):
    label_number = int(y_train[i, 0])
    label_name = class_names[label_number]

    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(f"{label_number}: {label_name}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 9. 클래스별 학습 이미지 개수 확인
# --------------------------------------------------
print("\n===== 클래스별 학습 이미지 개수 =====")

# (50000, 1) → (50000,)
labels_flat = y_train.flatten()

for class_number, class_name in enumerate(class_names):
    count = np.sum(labels_flat == class_number)

    print(
        f"{class_number}: "
        f"{class_name:10s} → {count}장"
    )