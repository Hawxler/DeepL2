# CIFAR10
# PyTorch를 numpy, sklearn 보다 앞서 임포트할 것
import torch
from torchvision.datasets import CIFAR10
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE = 42, 10, 64, 0.001
torch.manual_seed(SEED)

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()

data_dir = script_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# download=True: data_dir에 없으면 다운로드해
train_data = CIFAR10(root=str(data_dir), train=True, download=True)
test_data = CIFAR10(root=str(data_dir), train=False, download=True)

X_all_raw = train_data.data
X_test_raw = test_data.data
y_all = np.asarray(train_data.targets, dtype=np.int64)
y_test = np.asarray(test_data.targets, dtype=np.int64)
print(train_data.data.shape)   # (50000, 32, 32, 3)
print(test_data.data.shape)    # (10000, 32, 32, 3)
print(y_all[:4], y_all.shape)  # [6 9 9 4] (50000,)
print(y_test[:4], y_all.shape) # [3 8 8 0] (50000,)

class_names = np.array([
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck",
])

print("전체 학습 이미지:", X_all_raw.shape)
print("첫 이미지:", X_all_raw[0].shape)
print("첫 정답:", y_all[0], class_names[y_all[0]])

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_all_raw, y_all,
    test_size=0.1, random_state=SEED, stratify=y_all,
)

# 정규화
def preprocess_images(images):
    return np.asarray(images, dtype=np.float32) / 255.0

X_train = preprocess_images(X_train_raw)
X_val = preprocess_images(X_val_raw)
X_test = preprocess_images(X_test_raw)

print("학습:", X_train.shape, y_train.shape)
print("검증:", X_val.shape, y_val.shape)
print("테스트:", X_test.shape, y_test.shape)
# 학습: (45000, 32, 32, 3) (45000,)
# 검증: (5000, 32, 32, 3) (5000,)
# 테스트: (10000, 32, 32, 3) (10000,)