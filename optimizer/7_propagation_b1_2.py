# 7_propagation_f1.py에 대응하는 역전파 코드
# Pytorch 코드: Python 3.11 필요
import torch
from torch import nn

# --------------------------------------------------
# 1. 입력과 정답
# --------------------------------------------------
x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
y = torch.tensor([[1.0]], dtype=torch.float64)

# --------------------------------------------------
# 2. 모델: 입력 2개 → 은닉 뉴런 2개 → 출력 1개
# --------------------------------------------------
model = nn.Sequential(
    nn.Linear(2, 2, bias=False),
    nn.Sigmoid(),
    nn.Linear(2, 1, bias=False),
    nn.Sigmoid(),
).double()

# PyTorch의 가중치 배열: (출력 뉴런 수, 입력 수)
#
# 은닉층:
# [[w1, w3],
#  [w2, w4]]
#
# 출력층:
# [[w5, w6]]
#
# 초기값을 지정할 때는 미분 기록이 필요하지 않음
with torch.no_grad():
    model[0].weight.copy_(
        torch.tensor([
            [0.3, 0.2],
            [0.2, 0.3],
        ], dtype=torch.float64)
    )

    model[2].weight.copy_(
        torch.tensor([
            [0.4, 0.5],
        ], dtype=torch.float64)
    )

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
)


def print_weights():
    hidden = model[0].weight.detach()
    output = model[2].weight.detach()

    values = [
        hidden[0, 0],  # w1
        hidden[1, 0],  # w2
        hidden[0, 1],  # w3
        hidden[1, 1],  # w4
        output[0, 0],  # w5
        output[0, 1],  # w6
    ]

    for i, value in enumerate(values, start=1):
        print(f"w{i} = {value.item():.6f}")


# --------------------------------------------------
# 3. 이전 기울기 초기화
# 가중치를 초기화하는 것이 아님
# --------------------------------------------------
optimizer.zero_grad()

# --------------------------------------------------
# 4. 순전파와 비용 계산
# PyTorch가 미분을 위한 계산 과정을 자동 기록
# --------------------------------------------------
pred_before = model(x)
loss_before = loss_fn(pred_before, y)

# --------------------------------------------------
# 5. 역전파: 기울기 계산 → 각 weight.grad에 저장
# 이 줄에서는 가중치가 바뀌지 않음
# --------------------------------------------------
loss_before.backward()

gh = model[0].weight.grad
go = model[2].weight.grad

gradient_values = [
    gh[0, 0], gh[1, 0],
    gh[0, 1], gh[1, 1],
    go[0, 0], go[0, 1],
]

print("=== 역전파로 구한 기울기 ===")
for i, value in enumerate(gradient_values, start=1):
    print(f"dC/dw{i} = {value.item():.6f}")

# --------------------------------------------------
# 6. 옵티마이저가 가중치 갱신
# --------------------------------------------------
optimizer.step()

print("\n=== 갱신된 가중치 ===")
print_weights()

# --------------------------------------------------
# 7. 갱신된 가중치로 다시 순전파 및 비용 계산
# 결과 확인만 하므로 미분 기록을 끔
# --------------------------------------------------
with torch.no_grad():
    pred_after = model(x)
    loss_after = loss_fn(pred_after, y)

print("\n=== 학습 전후 비교 ===")
print(
    f"예측값: {pred_before.item():.6f}"
    f" → {pred_after.item():.6f}"
)
print(
    f"비용:   {loss_before.item():.6f}"
    f" → {loss_after.item():.6f}"
)