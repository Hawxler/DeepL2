# Cross Entropy Error
# 정답 클래스에 높은 확률을 줄수록 교차 엔트로피 손실이 작아지는지 보기
# 이를 위해 확률을 for문으로 직접 바꿔보며 정답일 확률의 놀람값을 손실로 계산해보고 그 변화를 느껴보는 코드.

import numpy as np
import matplotlib.pylab as plt

def cross_entropy_error(y, t):
    delta = 1e-7   # log(0)이 되어 무한대가 안 되게 작은 값 추가
    loss = -np.sum(t * np.log(y + delta))
    return loss

# 정답 (one-hot encoding으로 하나가 선별됨)
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) 

# 2. 그래프로 보기 (y_val[2] val값만 0~1까지 변화시킴. t[2]=1 정답이므로.)
x_axis = np.arange(0.0, 1.0, 0.01)
y_axis = []

for val in x_axis:
    # 10개 클래스에 대한 모델의 예측 확률 분포값(총합 = 1.0)
    # val 외의 값도 변수처리해야 val의 변화에 따라 나머지 클래스의 확률도 변하지만 시범 데이터이므로 무작위 수치를 분포시킴.
    y_val = np.array([0.05, 0.05, val, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    y_axis.append(cross_entropy_error(y_val, t))

plt.plot(x_axis, y_axis)
plt.xlabel('Predicted Probability')
plt.ylabel('CEE')
plt.title("CEE vs. Predicted Probability")
plt.show()
