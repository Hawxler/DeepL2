# 여러번 반복해보기
def sgd(param, grad, lr=0.1):
    return param - lr*grad

w = 10.0
learning_rate = 0.1

for step in range(5):
    grad = 2 * w  #예: loss=w^2라고 가정한 미분값
    w = sgd(w, grad, learning_rate)
    print(f"{step+1}회 후 w = {w:.4f}")
    