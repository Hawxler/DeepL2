# 기본형: 간단한 선형 회귀 예제
import numpy as np
from sklearn.linear_model import LinearRegression

# 입력 데이터: 공부 시간
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

# 정답 데이터: 시험점수
y = np.array([50, 61, 70, 79, 90])

# 모델 생성
model1 = LinearRegression()

# 학습
model1.fit(X, y)

# 예측: 6시간 공부 시 점수 예측
pred1 = model1.predict([[6]])

print("예측 점수:", pred1[0])
print("기울기:", model1.coef_[0])
print("절편:", model1.intercept_)
