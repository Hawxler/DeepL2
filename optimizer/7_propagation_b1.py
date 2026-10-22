# 7_propagation_f1.py에 대응하는 역전파 코드

import math

def sigmoid(n):
    return 1.0 / (1.0 + math.exp(-n))


# --------------------------------------------------
# 1. 입력, 정답, 초기 가중치
# --------------------------------------------------
x1 = 1.0
x2 = 2.0
y = 1.0

learning_rate = 0.1

weights = {
    "w1": 0.3,  # x1 → 첫 번째 은닉 뉴런
    "w2": 0.2,  # x1 → 두 번째 은닉 뉴런
    "w3": 0.2,  # x2 → 첫 번째 은닉 뉴런
    "w4": 0.3,  # x2 → 두 번째 은닉 뉴런
    "w5": 0.4,  # 첫 번째 은닉 뉴런 → 출력 뉴런
    "w6": 0.5,  # 두 번째 은닉 뉴런 → 출력 뉴런
}


# --------------------------------------------------
# 2. 순전파
# --------------------------------------------------
def forward(x1, x2, w):
    # 첫 번째 은닉 뉴런
    n1 = w["w1"] * x1 + w["w3"] * x2
    s1 = sigmoid(n1)

    # 두 번째 은닉 뉴런
    n2 = w["w2"] * x1 + w["w4"] * x2
    s2 = sigmoid(n2)

    # 출력 뉴런
    n3 = w["w5"] * s1 + w["w6"] * s2
    s3 = sigmoid(n3)

    # 역전파에서 사용할 중간값도 반환
    return {
        "n1": n1, "s1": s1,
        "n2": n2, "s2": s2,
        "n3": n3, "s3": s3,
    }


# --------------------------------------------------
# 3. 비용 계산
# --------------------------------------------------
def calculate_cost(y, s3):
    return (y - s3) ** 2


# --------------------------------------------------
# 4. 역전파: 가중치 6개의 기울기 계산
#    이 함수 안에서는 가중치를 갱신하지 않습니다.
# --------------------------------------------------
def backward(x1, x2, y, w, values):
    s1 = values["s1"]
    s2 = values["s2"]
    s3 = values["s3"]

    # 비용 → 출력값 → 출력 뉴런 가중합
    dC_ds3 = -2.0 * (y - s3)
    ds3_dn3 = s3 * (1.0 - s3)

    dC_dn3 = dC_ds3 * ds3_dn3

    # 출력층 가중치의 기울기
    # ∂n3/∂w5 = s1
    # ∂n3/∂w6 = s2
    dC_dw5 = dC_dn3 * s1
    dC_dw6 = dC_dn3 * s2

    # 첫 번째 은닉 뉴런으로 역전파
    # 여기서는 갱신 전 w5를 사용
    dn3_ds1 = w["w5"]
    ds1_dn1 = s1 * (1.0 - s1)

    dC_dn1 = dC_dn3 * dn3_ds1 * ds1_dn1

    dC_dw1 = dC_dn1 * x1
    dC_dw3 = dC_dn1 * x2

    # 두 번째 은닉 뉴런으로 역전파
    # 여기서는 갱신 전 w6를 사용
    dn3_ds2 = w["w6"]
    ds2_dn2 = s2 * (1.0 - s2)

    dC_dn2 = dC_dn3 * dn3_ds2 * ds2_dn2

    dC_dw2 = dC_dn2 * x1
    dC_dw4 = dC_dn2 * x2

    return {
        "w1": dC_dw1,
        "w2": dC_dw2,
        "w3": dC_dw3,
        "w4": dC_dw4,
        "w5": dC_dw5,
        "w6": dC_dw6,
    }


# --------------------------------------------------
# 5. 경사하강법: 모든 기울기를 구한 후 가중치 갱신
# --------------------------------------------------
def update_weights(w, gradients, lr):
    return {
        name: w[name] - lr * gradients[name]
        for name in w
    }


# --------------------------------------------------
# 6. 학습 1회 실행
# --------------------------------------------------

# ① 기존 가중치로 순전파 및 비용 계산
before = forward(x1, x2, weights)
cost_before = calculate_cost(y, before["s3"])

print("=== 갱신 전 순전파 ===")
for name, value in before.items():
    print(f"{name} = {value:.6f}")
print(f"cost = {cost_before:.6f}")

# ② 기존 가중치와 순전파 결과로 모든 기울기 계산
gradients = backward(x1, x2, y, weights, before)

print("\n=== 역전파: 가중치별 기울기 ===")
for name, gradient in gradients.items():
    print(f"∂C/∂{name} = {gradient:.6f}")

# ③ 가중치 6개 갱신
new_weights = update_weights(
    weights, gradients, learning_rate
)

print("\n=== 가중치 갱신 ===")
for name in weights:
    old = weights[name]
    gradient = gradients[name]
    new = new_weights[name]

    print(
        f"{name}: {old:.6f}"
        f" - {learning_rate} × ({gradient:.6f})"
        f" = {new:.6f}"
    )

# ④ 새 가중치로 순전파 및 비용 재계산
after = forward(x1, x2, new_weights)
cost_after = calculate_cost(y, after["s3"])

print("\n=== 갱신 후 순전파 ===")
for name, value in after.items():
    print(f"{name} = {value:.6f}")
print(f"cost = {cost_after:.6f}")

print("\n=== 학습 전후 비교 ===")
print(f"예측값: {before['s3']:.6f} → {after['s3']:.6f}")
print(f"비용:   {cost_before:.6f} → {cost_after:.6f}")
print(f"비용 감소량: {cost_before - cost_after:.6f}")