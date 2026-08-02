# 학습이 진행되면서 loss 감소 출력
import numpy as np

def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

x = np.array([1.0, 2.0])
t = np.array([0, 0, 1])

W = np.array([
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 0.3]
])

b = np.array([0.0, 0.0, 0.0])

learning_rate = 0.1

for step in range(20):
    z = x @ W + b
    y = softmax(z)
    loss = cross_entropy_error(y, t)

    dz = y - t
    dW = np.outer(x, dz)
    db = dz

    W -= learning_rate * dW
    b -= learning_rate * db

    print(
        f"step {step:02d} | "
        f"정답 클래스 확률 y[2]: {y[2]:.4f} | "
        f"loss: {loss:.4f}"
    )