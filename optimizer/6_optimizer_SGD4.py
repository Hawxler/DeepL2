# w, b 둘 다 업데이트하는 예
# 간단한 데이터
x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]  #y = 2x + 1

w = 0.0
b = 0.0
lr = 0.1

for epoch in range(20):
    dw = 0.0
    db = 0.0
    loss = 0.0

    # 전체 데이터에 대해 loss와 gradient 계산
    for x, y in zip(x_data, y_data):
        pred = w * x + b
        error = pred - y

        loss += error ** 2  # MES의 합 형태
        dw += 2 * error * x # d/dw: error**2*x->2error*x
        db += 2 * error     # d/db: error**2+b->2error
    
    # 평균
    n = len(x_data)
    loss /= n
    dw /= n
    db /= n

    # SGD 업데이트
    w = w - lr * dw
    b = b - lr * db

    print(f"epoch={epoch+1:2d}, loss={loss:.4f},",
          f"w={w:.4f}, b={b:.4f}")