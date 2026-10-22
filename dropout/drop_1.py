# Dropout을 파이썬으로 구현함.
import random


def dropout(values, p=0.5, training=True):
    # 이 예제에서는 p=1을 제외하여 0으로 나누는 것을 방지
    if not 0 <= p < 1:
        raise ValueError("p는 0 이상 1 미만이어야 합니다.")

    # 추론 모드에서는 그대로 통과
    if not training:
        return values.copy()

    result = []
    for value in values:
        if random.random() < p:
            result.append(0.0)
        else:
            result.append(value / (1 - p))

    return result


x = [2.0, 4.0, 6.0, 8.0]

# 재현을 위해 한 번만 설정
random.seed(1)

print("입력:", x)
print("학습(Dropout을 거친 값):", dropout(x, p=0.5, training=True))
print("추론(Dropout을 거치지 않은 값):", dropout(x, p=0.5, training=False))