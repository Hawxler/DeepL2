# Cross Entropy Error
import numpy as np
import matplotlib.pylab as plt

def cross_entropy_error(y, t):
    delta = 1e-7 # log(0)가 되어 무한대가 안 되도록 매우 작은 값 추가
    return -np.sum(t * np.log(y + delta)) # CEE = - Σ t_i log(y_i)

# 정답 (one-hot encoding으로 하나가 선별됨)
t = np.zeros(10)
t[2] = 1  # [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]

x_axis = np.arange(0.0, 1.0, 0.01)
y_axis = []

for val in x_axis:
    y_val = np.array([0.05, 0.05, val, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    y_axis.append(cross_entropy_error(y_val, t))

print([f'{x:.4f}' for x in y_axis])

plt.plot(x_axis, y_axis)
plt.xlabel("Predicted Probability")
plt.ylabel('CEE')
plt.title("CEE vs. Predicted Probability")
plt.show()

####[참고]##################################
# 10개 클래스에 대한 모델의 예측 확률 분포값(총합 = 1.0)
y = np.array([0.05, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
y = y / y.sum()  # 전체 합을 1로 정규화(y[0]/0.95, y[1]/0.95 ...)
print("SUM:", y.sum())

# 1. Cross Entropy Error 테스팅
res = cross_entropy_error(np.array(y), np.array(t))
print(f"{res:.2f}")   # 0.46
############################################