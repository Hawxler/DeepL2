# 클래스로 좀 더 옵티마이져답게 만들기
class SimpleSGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update(self, param, grad):
        return param - self.learning_rate * grad


optimizer = SimpleSGD(learning_rate=0.1)

w = 10.0

for step in range(5):
    grad = 2 * w   # loss = w^2 의 미분
    w = optimizer.update(w, grad)
    print(f"{step+1}회 후 w = {w:.4f}")