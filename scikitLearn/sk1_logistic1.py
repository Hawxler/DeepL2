# 기본형: 간단한 분류 모델 예제
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#1. 데이터 불러오기
iris = load_iris()

X = iris.data
y = iris.target

'''
print(X)
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 ...
 [6.2 3.4 5.4 2.3]
 [5.9 3.  5.1 1.8]]

print(y)
[0 0 0 0 ... 1 1 1 1 ... 2 2 2 2 ...]
 '''
# 2. 데이터 분리: 학습용, 테스트용
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape) # (120, 4)
print("X_test:", X_test.shape)   # (30, 4)
print("y_train:", y_train.shape) # (120,)
print("y_test:", y_test.shape)   # (30,)

# 3. 모델 생성
model1 = LogisticRegression(max_iter=200)

# 4. 학습
model1.fit(X_train, y_train)

# 5. 예측
y_pred = model1.predict(X_test)

# 6. 정확도 평가
acc = accuracy_score(y_test, y_pred)

print("실제 정답:", y_test)
print("예측 결과:", y_pred)
print("정확도:", acc)

print(iris.feature_names) 
# ['sepal length (cm)', 'sepal width (cm)', 
# 'petal length (cm)', 'petal width (cm)']
