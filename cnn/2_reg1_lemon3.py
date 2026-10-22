# Pytorch, Forward 직접 작성 방식
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

# 1. 데이터 준비
url = "https://raw.githubusercontent.com/hawklux/DeepLearn/main/1_Data/lemonade.csv"
data = pd.read_csv(url)
print(data.head())

# 입력: 온도 / 정답: 판매량
inVar = data[['온도']].to_numpy(dtype=np.float32)
tarVar = data[['판매량']].to_numpy(dtype=np.float32)

# Numpy 배열 -> PyTorch 텐서
x_train = torch.from_numpy(inVar)
y_train = torch.from_numpy(tarVar)

print("입력 형태:", x_train.shape)  # (샘플 수: 1)
print("정답 형태:", y_train.shape)  # (샘플 수: 1)

# 입력과 정답을 쌍으로 묶고 배치 구성
dataset = TensorDataset(x_train, y_train)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 모델 구조: nn.Module + forward() -> 순전파 직접 작성
class Model1(nn.Module):
    def __init__(self):
        super().__init__()

        # Activation: 입력 특성 1개 -> 출력값 1개
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # 순전파: y_hat = wx + b
        y_hat = self.linear(x)
        return y_hat
    
# 모델 객체 생성
model1 = Model1()

# 3. 손실 함수와 옵티마이저 설정
loss_fn = nn.MSELoss()

optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001,)

# 4. 모델 학습
model1.train()  # 학습 모드 설정

for epoch in range(1000):
    total_loss = 0.0

    for batch_x, batch_y in loader:
        # (1) 이전 (batch의 역전파용) 기울기 초기화
        optimizer1.zero_grad()

        # (2) 순전파: 판매량 예측
        predictions = model1(batch_x)

        # (3) 손실 계산: 실제 판매량 - 예측 판매량
        loss = loss_fn(predictions, batch_y)

        # (4) 역전파: 가중치와 편향의 기울기 계산
        loss.backward()

        # (5) 가중치와 편향의 업데이트
        optimizer1.step()

        # (6) 전체 epoch의 평균 손실을 구하기 위해 누적
        total_loss += loss.item() * batch_x.size(0)

    # 100 epoch마다 학습 손실 출력
    if (epoch + 1) % 100 == 0:
        average_loss = total_loss / len(dataset)
        print(
            f"Epoch {epoch + 1:4d}/1000, "
            f"Loss: {average_loss:.4f}"
        )
# 5. 모델 이용 예측
# 샘플 1개, 특성 1개 -> shape: (1, 1)
x_new = torch.tensor([[15.0]], dtype=torch.float32)

model1.eval()  # 평가, 추론 모드 설정

with torch.no_grad():  # 추론이므로 역전파 안 함. 학습이 완료된 가중치를 그대로 사용.
    pred = model1(x_new).item()
print(f"15도일 때 예상 판매량: {pred:.2f}")

# 6. 학습된 가중치와 편향 확인
w = model1.linear.weight.item()
b = model1.linear.bias.item()

print(f"가중치 w: {w:.4f}")
print(f"편향 b: {b:.4f}")
print(f"학습된 식: 예상 판매량 = {w:.4f} x 온도 + {b:.4f}")