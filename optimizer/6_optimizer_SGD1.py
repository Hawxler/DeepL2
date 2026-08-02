# 버전 1: 간단한 SGD optimizer 흉내
def sgd(param, grad, lr=0.1):
    return param - lr*grad

#예시
w = 10.0
grad = 2.5  # 손실률 w로 미분한 값이라고 가정함.
learning_rate = 0.1

new_w = sgd(w, grad, learning_rate)

print("업데이트 전 w:", w)
print("grad:", grad)
print("업데이트 후 w:", new_w)