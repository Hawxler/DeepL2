# PyTorch Sequencial 코드
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

# 1. 데이터 준비
url = "https://raw.githubusercontent.com/hawklux/DeepLearn/main/1_Data/lemonade.csv"
data = pd.read_csv(url)
print(data.head())

# 입력: 온도 / 정답: 판매량
x_features = data[['온도']].to_numpy(dtype=np.float32)
y_target = data[['판매량']].to_numpy(dtype=np.float32)

# Numpy 배열 -> PyTorch 텐서
x_train = torch.from_numpy(x_features)
y_train = torch.from_numpy(y_target)

print("입력 형태:", x_train.shape)  #(샘플 수, 특성 칼럼 수 n)
print("정답 형태:", y_train.shape)  #(샘플 수, 타겟 칼럼 수 1)

# 입력과 정답을 묶고 배치 구성
dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True,)

# 2. 모델 구조 구축: Sequential 방식
model1 = nn.Sequential(
    nn.Linear(1, 1),  # 입력 특성 1개 -> 출력값 1개
)

# 손실 함수와 옵티마이저
loss_fn = nn.MSELoss()

optim1 = torch.optim.Adam(model1.parameters(), lr=0.001,)

print(model1)

# 3. 모델 학습
model1.train()  # 학습 모드로 전환 (학습 모드 <-> 추론 모드)

for epoch in range(1000):
    for batch_x, batch_y in loader:
        # (1) 이전 batch 기울기 초기화
        optim1.zero_grad()

        # (2) 순전파: 판매량 예측
        prediction = model1(batch_x)

        # (3) 손실 계산
        loss = loss_fn(prediction, batch_y)

        # (4) 역전파: 기울기 계산
        loss.backward()

        # (5) 가중치/편향 업데이트
        optim1.step()

# 4. 모델 이용 예측
# 샘플 1개, 특성 1개 -> shape: (1, 1)
x_new = torch.tensor([[15.0]], dtype=torch.float32)

model1.eval()  # 추론 모드로 전환(학습 모드 <-> 추론 모드)

with torch.no_grad():
    pred = model1(x_new).item()

print(f"15도일 대 예상 판매량: {pred:.2f}")

# 5. 학습된 가중치와 편향 확인
# model1[0]은 Linear 층
w = model1[0].weight.item()
b = model1[0].bias.item()

print(f"가중치 w: {w:.4f}")
print(f"편향 b: {b:.4f}")