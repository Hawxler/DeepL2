def relu(z):
    return max(0.0, z)

def forward(x1, x2):
    # 은닉층: 뉴런 2개
    z1 = 0.5 * x1 + 1.0 * x2 - 0.5
    h1 = relu(z1)
    z2 = -1.0 * x1 + 0.25 * x2 - 0.5
    h2 = relu(z2)
 
    # 출력층: 뉴런 1개, 선형 출력
    y_pred = 1.5 * h1 + (-1.0) * h2 + 0.5
    return z1, h1, z2, h2, y_pred

# 1. 순전파: 입력만으로 예측값 계산
x1, x2 = 1.0, 2.0
z1, h1, z2, h2, y_pred = forward(x1, x2)
print(f"은닉 뉴런 1: z1={z1:.1f}, h1={h1:.1f}")
print(f"은닉 뉴런 2: z2={z2:.1f}, h2={h2:.1f}")
print(f"예측값: {y_pred:.1f}")

# 2. 손실 계산: 예측값을 정답과 비교
# 샘플 1개, 출력값 1개이므로 제곱 오차가 곧 MSE
y_true = 4.0
loss = (y_pred - y_true) ** 2
print(f"정답: {y_true:.1f}")
print(f"손실(MSE): {loss:.2f}")