# PyTorch Sequential API 코드

######################################
# 1. 데이터 준비(0_dataPrep1.py의 내용)
######################################
# 데이터 준비: 불러오기 > 입력/정답 분리 > train_test_split > 입력 표준화(StandardScaler) > 정답 One-hot encoding
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 읽기
url = "https://raw.githubusercontent.com/hawklux/DeepLearn/refs/heads/main/0_Data/bong/iris.csv"
data = pd.read_csv(url)

print(data.shape)
print(data.head())
print(data.columns)

# 2. 입력 특성
feature_columns = ['꽃받침길이', '꽃받침폭', '꽃잎길이', '꽃잎폭']

X = data[feature_columns].to_numpy(dtype=np.float32)

# 3. 정답의 클래스 순서를 고정
class_names = np.array(['setosa', 'versicolor', 'virginica'])
class_to_index = {
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2,
}

y_index = data['품종'].map(class_to_index)
if y_index.isna().any():
    raise ValueError("품종 값 오류")

# 4. 데이터 Shuffle + 분할: 80% 학습, 20% 테스트
X_train_raw, X_test_raw, y_train_index, y_test_index = train_test_split(
    X,
    y_index,
    test_size=0.2,
    random_state=42,
    stratify=y_index,
)

# 5. 입력 표준화: 학습 데이터로만 평균과 표준편차 계산
# fit_transform(훈련 데이터): fit(평균, 표준편차 구하기) + transform(변환)
# transform(테스트 데이터): 앞서 구해둔 동일한 기준(평균, 표준편차) 적용해서 변환
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test = scaler.transform(X_test_raw).astype(np.float32)
print(f"학습용 입력: {X_train.shape}, {X_train[:1]}" )
print(f"테스트용 입력: {X_test.shape}, {X_test[:1]}")

# 6. 출력 원-핫 인코딩
# 0 -> [1, 0, 0], 1 -> [0, 1, 0], 2 -> [0, 0, 1]
y_train = np.eye(3, dtype=np.float32)[y_train_index]
y_test = np.eye(3, dtype=np.float32)[y_test_index]

print(f"학습용 정답: {y_train.shape}, {y_train[0]}") #(120, 3), [1. 0. 0.]
print(f"테스트용 정답: {y_test.shape}, {y_test[:1]}") #(30, 3), [[1. 0. 0.]]
print("섞인 행 인덱스, 품종 인덱스:\n", y_train_index[:5])
print("One-hot 정답:\n", y_train[:5])

# 7. 추론 연습용 데이터 스케일링
# 순서: 꽃받침길이, 꽃받침폭, 꽃잎길이, 꽃잎폭
X_new_raw = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.5, 3.0, 5.8, 2.2]
], dtype=np.float32)

# Scaling 해서 X_new 만들기
# transform(): 추론도 학습 때 구한 동일한 기준(평균, 표준편차) 사용
X_new = scaler.transform(X_new_raw).astype(np.float32)
print(X_new[:2])

######################################
# 2. 모델 구축(PyTorch Sequential API)
# python 3.10 가상환경에 pytorch 설치
#  .\myenv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
######################################
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.BatchNorm1d(16),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(16, 8),
    nn.BatchNorm1d(8),
    nn.ReLU(),

    # Softmax 전 로짓 3개
    nn.Linear(8, 3),
)

print(model)

#################################################
# 3. 학습
#################################################
# 3.1 Numpy -> Tensor
x_train_tensor = torch.from_numpy(X_train) # (120, 4): [5.1, 3.5, 1.4, 0.2]
y_train_tensor = torch.from_numpy(y_train) #원-핫 (120, 3): [1.0, 0.0, 0.0]

x_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)
x_new_tensor = torch.from_numpy(X_new)

# 3.2 배치 준비
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
)
# DataLoader(): 데이터셋을 shuffle하고 모델이 학습하기 좋은 크기로 batch로 묶어서 for문으로 한 묶음씩 계속 공급해줌.
# print(train_dataset[0]) #(tensor([-1.7216, -0.3248, -1.3470, -1.3202]), tensor([1., 0., 0.]))

# 3.3 손실함수와 옵티마이저 설정
# CrossEntropyloss는 로짓과 실수 원-핫 정답도 받을 수 있음
loss_fn = nn.CrossEntropyLoss()  # 로짓을 받아 로그 확률 도출 = 출력 레이어에서 Softmax 없어도 argmax로 정답 도출
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3.4 학습
for epoch in range(200):
    model.train()
    total_loss = 0.0
    correct = 0
    sample_count = 0

    # 배치별 계산: 크기 최대 16, 손실-기울기 계산 > 파라미터 업데이트
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()  # 배치마다 미분값 초기화: w_new = (w_cur 1.0) - (학습률 0.1 * 미분값 2.0) = 1.0 - 0.2 = 0.8 -> 가중치 0.8은 다음 배치로 넘김, 미분값 2.0은 지우고(0) 배치마다 다시 계산하게 함.
        logits = model(batch_x)          # 순전파: 활성함수에 넣을 로짓
        loss = loss_fn(logits, batch_y)  # 손실 = -log(정답 클래스의 예측 확률)
        loss.backward()                  # 기울기 계산만 함(적용은 step()가 함)
        optimizer.step() # 역전파로 계산된 기울기를 바탕으로 최적화 규칙(Adam)에 따라 모델의 가중치와 편향을 갱신함.

        batch_size = batch_x.size(0) # 텐서의 0번(현재 배치)의 차원 크기
        total_loss += loss.item() * batch_size #loss.item(): loss 텐서에서 값 꺼냄
        # 추후: loss 계산 = 배치 평균 손실 = total_loss / sample_count

        predicted_indices = logits.argmax(dim=1) # 예측 정답: 가장 큰 값이 있는 위치의 인덱스
        true_indices = batch_y.argmax(dim=1) # 진짜 정답: [1.0, 0.0, 0.0] -> 0: 원-핫 중 정답 인덱스
        # (예측 정답, 진짜 정답 비교).sum() > 맞은 개수: 추후 정확도(Accuracy) 계산
        correct += (predicted_indices == true_indices).sum().item()
        sample_count += batch_size
    
    if (epoch + 1) % 20 == 0:
        average_loss = total_loss / sample_count  #모니터링용 평균손실(계산용 아님)
        accuracy = correct / sample_count         #모니터링용 정확도 = keras에서 model.compile(matrix=['accuracy'])와 동일한 평가지표
        print(
            f"Epoch {epoch + 1:3d} / 200 | "
            f"loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.2%}"
        )

#################################################
# 4. 평가 > 추론: Sequential/Functinal 동일
#################################################
# 4.1 평가
model.eval()  # model.train() 모드에서 .eval() 모드로 전환
# .eval() 모드에서는: 
# - dropout(): 꺼서 모든 뉴런 사용
# - BatchNorm: 꺼서 모든 데이터의 누적 평균과 분산 사용

with torch.no_grad():
    test_logits = model(x_test_tensor)
    test_loss = loss_fn(test_logits, y_test_tensor).item()

    predicted_indices = test_logits.argmax(dim=1) # dim=0: 행 비교, 1=열 비교 max
    true_indices = y_test_tensor.argmax(dim=1)

    test_accuracy = (
        (predicted_indices == true_indices).float().mean().item()
    )
    # predicted_indices == true_indices: [True, False, True, True] 불리언 텐서
    # .float(): [True, False, True, True] -> [1.0, 0.0, 1.0, 1.0]
    # .mean(): ([0.75]) Accuracy
    # .item(): 0.75

print(f"테스트 손실: {test_loss:.4f}")
print(f"테스트 정확도: {test_accuracy:.2%}")

# 4.2 추론
model.eval()  # model.train() 모드에서 .eval() 모드로 전환

with torch.no_grad():
    logits = model(x_new_tensor)

    # 추론에서 로짓 -> 확률 변환
    probabilities = torch.softmax(logits, dim=1)
    predicted_indices = probabilities.argmax(dim=1)

for i, class_index in enumerate(predicted_indices.tolist()):
    print(
        f"입력: {X_new_raw[i]} | "
        f"예측 품종: {class_names[class_index]} | "
        f"확률: {np.round(probabilities[i].numpy(), 4)}"
    )