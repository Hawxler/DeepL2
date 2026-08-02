class SimpleAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0.0
        self.v = 0.0
        self.t = 0

    def update(self, param, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        param = param - self.lr * m_hat / ((v_hat ** 0.5) + self.epsilon)
        return param
    
adam1 = SimpleAdam(lr=0.1)
w = 10

for step in range(10):
    grad = 2 * w    # loss = w**2
    w = adam1.update(w, grad)
    print(f"{step+1}회 후 w = {w:.6f}")