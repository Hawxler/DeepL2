from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 0) 예시 데이터 생성 (사용하는 X, y로 대체하면 됨)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

print(len(X)) # 1000
print(X[0]) # [...]
print(len(X[0])) # 20
print(len(y)) # 1000
print(y[:4]) # [1 0 1 1]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(len(X_train))
print(X_train[0])
print(y_train[:4])

print(len(X_val))
print(y_test[:4])