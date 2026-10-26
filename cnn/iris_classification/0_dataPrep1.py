# 데이터 준비: 불러오기 > 입력/정답 분리 > train_test_split > 입력 표준화(StandardScaler) > 정답 One-hot encoding
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 읽기
url = "https://raw.githubusercontent.com/hawklux/DeepLearn/refs/heads/main/0_Data/bong/iris.csv"
data = pd.read_csv(url)

print(data.shape)
print(data.head())
print(data.columns)

# 2. 입력 특성
feature_columns = ['꽃받침길이', '꽃받침폭', '꽃잎길이', '꽃잎폭']

X = data[feature_columns].to_numpy(dtype=np.float32)

# 3. 정답의 클래스 순서를 고정
class_names = np.array(['setosa', 'versicolor', 'virginica'])
class_to_index = {
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2,
}

y_index = data['품종'].map(class_to_index)
if y_index.isna().any():
    raise ValueError("품종 값 오류")

# 4. 데이터 Shuffle + 분할: 80% 학습, 20% 테스트
X_train_raw, X_test_raw, y_train_index, y_test_index = train_test_split(
    X,
    y_index,
    test_size=0.2,
    random_state=42,
    stratify=y_index,
)

# 5. 입력 표준화: 학습 데이터로만 평균과 표준편차 계산
# fit_transform(훈련 데이터): fit(평균, 표준편차 구하기) + transform(변환)
# transform(테스트 데이터): 앞서 구해둔 동일한 기준(평균, 표준편차) 적용해서 변환
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test = scaler.transform(X_test_raw).astype(np.float32)
print(f"학습용 입력: {X_train.shape}, {X_train[:1]}" )
print(f"테스트용 입력: {X_test.shape}, {X_test[:1]}")

# 6. 출력 원-핫 인코딩
# 0 -> [1, 0, 0], 1 -> [0, 1, 0], 2 -> [0, 0, 1]
y_train = np.eye(3, dtype=np.float32)[y_train_index]
y_test = np.eye(3, dtype=np.float32)[y_test_index]

print(f"학습용 정답: {y_train.shape}, {y_train[0]}") #(120, 3), [1. 0. 0.]
print(f"테스트용 정답: {y_test.shape}, {y_test[:1]}") #(30, 3), [[1. 0. 0.]]
print("섞인 행 인덱스, 품종 인덱스:\n", y_train_index[:5])
print("One-hot 정답:\n", y_train[:5])

# 7. 추론 연습용 데이터 스케일링
# 순서: 꽃받침길이, 꽃받침폭, 꽃잎길이, 꽃잎폭
X_new_raw = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.5, 3.0, 5.8, 2.2]
], dtype=np.float32)

# Scaling 해서 X_new 만들기
# transform(): 추론도 학습 때 구한 동일한 기준(평균, 표준편차) 사용
X_new = scaler.transform(X_new_raw).astype(np.float32)
print(X_new[:2])