# PyTorch Sequential 코드

#########################################
#1. 이미지 전처리 (0_DataPrep1.py 내용)
# MNIST 데이터셋 전처리
# 28 x 28 흑백, 학습용: 60,000, 테스트용 10,000, 검증용 6,000
# 현실 이미지 경우 예: 648x480 -> 패딩추가-> 648x648 -> 224x224
#########################################

# 원-핫 없이 정수 정답 0~9 사용해보기
import os
from urllib.request import urlretrieve

#numpy, sklearn 보다 앞서 임포트할 것
from torch.utils.data import TensorDataset, DataLoader 

import numpy as np
from sklearn.model_selection import train_test_split

SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE = 42, 5, 64, 0.001

# 1.1 데이터 다운로드 (현재 폴더 아래 data/mnist.npz)
try:
    # 일반 파이썬(.py) 파일로 실행할 때
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 주피터 노트북(Jupyter Notebook)이나 구글 코랩(Colab)에서 실행할 때는 __file__ 파일 경로가 없으로 아래처럼.
    script_dir = os.getcwd()

data_dir = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True) # data 폴더가 없으면 생성

data_path = os.path.join(data_dir, "mnist.npz")

if not os.path.exists(data_path): # data_path에 mnist.npz가 없다면
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    print("MNIST 다운 중...")
    urlretrieve(url, data_path)
    print("다운로드 완료!")

# 1.2 데이터 읽기(npz -> all + test)
with np.load(data_path, allow_pickle=False) as data: # 보안: 피클은 차단
    X_all_raw = data["x_train"]
    y_all = data['y_train']

    X_test_raw = data["x_test"]
    y_test = data["y_test"]

# 1.3 데이터 살펴보기 (전체 출력 대신 안전하게 shape과 샘플만 출력)
np.set_printoptions(linewidth=200)
print("X_all_raw shape:", X_all_raw.shape)  # (60000, 28, 28)
print("첫 번째 이미지 픽셀:\n", X_all_raw[:1]) # 5 이미지 픽셀
print("첫 번째 이미지의 9번째 픽셀값:", X_all_raw[0][8])
print("첫 번째 이미지 shape:", X_all_raw[0].shape) # (28, 28)
print("첫 번째 정답(y):", y_all[0])                # 5
print("y_all shape:", y_all.shape)                # (60000,)

# 1.4 학습 데이터에서 검증 데이터 10% 분리 (all -> train + val)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_all_raw,
    y_all,
    test_size=0.1,
    random_state=SEED,
    stratify=y_all, # 0~9 이미지 누락 없이 골고루 넣기
)

# 1.5 픽셀 값 정규화: 0~255 -> 0~1
def preprocess_images(images):
    return np.asarray(images, dtype=np.float32) / 255.0

X_train = preprocess_images(X_train_raw)
X_val = preprocess_images(X_val_raw)
X_test = preprocess_images(X_test_raw)

# 1.6 정답: 정수 클래스 번호
y_train = y_train.astype(np.int64)
y_val = y_val.astype(np.int64)
y_test = y_test.astype(np.int64)

print("학습:", X_train.shape, y_train.shape) # (54000, 28, 28) (54000,)
print("검증:", X_val.shape, y_val.shape)     # (6000, 28, 28) (6000,)
print("테스트:", X_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

######################################
# 2. 모델 구축(PyTorch Sequential API)
######################################
import torch
from torch import nn

torch.manual_seed(SEED)

model = nn.Sequential(
    # 입력 선언 불필요(Keras에서는 필요)
    # Conv2d(배치, 채널, 높이, 너피)
    nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    # 두 번째 합성곱 블록
    nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    # 완전연결층: 분류 실행
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 10),  # softmax에 넣을 logits
)

print(model)

######################################
# 3. 학습
# from torch.utils.data import TensorDataset, DataLoader
######################################

# 3.1 GPU 안 되면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("사용 장치:", device)

# 3.2 Numpy -> Tensor, 채널 축 추가
# (N, 28, 28) -> (N, 1, 28, 28)
def make_loader(images, labels, shuffle=False):
    x_tensor = torch.from_numpy(images).unsqueeze(1)
    y_tensor = torch.from_numpy(labels)
    dataset = TensorDataset(x_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_test, y_test)

# 3.3 손실 함수와 옵티마이저: logits + int64 정수 정답
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3.4 검증·테스트 공통 평가 함수
def evaluate_torch(model, loader):
    model.eval()  # model.train() 모드에서 .eval() 모드로 전환
    # .eval() 모드에서는: 
    # - dropout(): 꺼서 모든 뉴런 사용
    # - BatchNorm: 꺼서 모든 데이터의 누적 평균과 분산 사용
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x) # Torch 모델 출력은 logits
            loss = loss_fn(logits, batch_y)
            predicted = logits.argmax(dim=1)

            batch_count = batch_y.size(0)
            total_loss += loss.item() * batch_count
            correct += (predicted == batch_y).sum().item()
            total += batch_count
    return total_loss / total, correct / total

# 3.5 학습
for epoch in range(EPOCHS):
    # train() 모드로 전환
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()  # 이전 배치 기울기 초기화
        logits = model(batch_x)  # 순전파
        loss = loss_fn(logits, batch_y)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 갱신

        predicted = logits.argmax(dim=1)
        batch_count = batch_y.size(0)
        total_loss += loss.item() * batch_count
        correct += (predicted == batch_y).sum().item()
        total += batch_count

    train_loss = total_loss / total
    train_accuracy = correct / total
    val_loss, val_accuracy = evaluate_torch(model, val_loader)

    print(
        f'Epoch {epoch + 1}/{EPOCHS} | '
        f"loss={train_loss:.4f}, accuracy={train_accuracy:.2%} | "
        f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.2%}"
    )

######################################
# 4. 평가/추론
######################################
# 4.1 테스트 데이터로 최종 평가
test_loss, test_accuracy = evaluate_torch(model, test_loader)
print(f"테스트 손실: {test_loss:.4f}")
print(f"테스트 정확도: {test_accuracy:.2%}")

# 4.2 추론 함수
# 입력: 정규롸 전 이미지, shape(N, 28, 28), 값 범위 0~255
def predict_torch(model, raw_images):
    images = preprocess_images(raw_images)
    images = torch.from_numpy(images).unsqueeze(1).to(device) 
    # x = torch.from_numpy(images) # ([5, 28, 28])
    # x = x.unsqueeze(1) # 1채널 축을 (1)의 자리에 추가: ([5, 1, 28, 28]). unsqueeze()는 1채널만 추가함. 3채널은 Conv2d()인수로 지정.

    model.eval()
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    return predictions.cpu().numpy(), probabilities.cpu().numpy()

# 4.3 테스트 이미지 5장으로 추론 사용법 확인
predictions, probabilities = predict_torch(model, X_test_raw[:5])

for i, predicted in enumerate(predictions):
    confidence = probabilities[i, predicted]
    print(
        f"이미지 {i}: 정답={y_test[i]}, "
        f"예측={predicted}, 확률={confidence:.2%}"
    )