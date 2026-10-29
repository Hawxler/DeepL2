# MNIST 데이터셋 전처리
# 28 x 28 흑백, 학습용: 60,000, 테스트용 10,000, 검증용 6,000
# 현실 이미지 경우 예: 648x480 -> 패딩추가-> 648x648 -> 224x224

# 원-핫 없이 정수 정답 0~9 사용해보기
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# 1. 데이터 다운로드 (현재 디렉터리에 data/mnist.npz 다운로드)
script_dir = Path(__file__).resolve().parent
data_path = script_dir / "data" / "mnist.npz"
data_path.parent.mkdir(parents=True, exist_ok=True) # 부모 + data_path

if not data_path.exists():  # data_path에 mnist.npz 파일이 없다면
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    print("MNIST 다운로드 중...")
    urlretrieve(url, data_path)  # 다운로드 실행

# 2. 데이터 읽기(npz -> all + test)
with np.load(data_path, allow_pickle=False) as data: # 보안: 피클 차단
    X_all_raw = data["x_train"]
    y_all = data['y_train']

    X_test_raw = data["x_test"]
    y_test = data['y_test']

np.set_printoptions(linewidth=200, threshold=np.inf) # 줄바꿈 없이 출력
print(X_all_raw[:1])       # 5 이미지 픽셀
print(X_all_raw.shape)     # (60000, 28, 28)
print(X_all_raw[0].shape)  # (28, 28)
print(y_all[0])            # 5
print(y_all.shape)         # (60000,)

# 3. 학습 데이터에서 검증 데이터 10% 분리 (all -> train + val)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_all_raw,
    y_all,
    test_size=0.1,
    random_state=SEED,
    stratify=y_all,
)

# 4. 픽셀 값 정규화: 0~255 -> 0~1
def preprocess_images(images):
    return np.asarray(images, dtype=np.float32) / 255.0

X_train = preprocess_images(X_train_raw)
X_val = preprocess_images(X_val_raw)
X_test = preprocess_images(X_test_raw)

# 5. 정답: 정수 클래스 번호
y_train = y_train.astype(np.int64)
y_val = y_val.astype(np.int64)
y_test = y_test.astype(np.int64)

print("학습:", X_train.shape, y_train.shape) # (54000, 28, 28) (54000,)
print("검증:", X_val.shape, y_val.shape)     # (6000, 28, 28) (6000,)
print("테스트:", X_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

#########################################
# 이미지 확인
#########################################
import matplotlib.pyplot as plt

#1. 첫 번째 이미지 + 정답 가져오기
image_to_show = X_all_raw[0]  # 28x28 2차원 배열
label_to_show = y_all[0]

print(f"이미지 레이블: {label_to_show}")

#2. 이미지 출력하기
plt.figure(figsize=(4, 4))
plt.imshow(image_to_show, cmap='gray')
plt.title(f"Label: {label_to_show}", fontsize=14)
plt.axis('off')
plt.show()