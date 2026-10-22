# 군집화(Clustering)
import numpy as np
from sklearn.cluster import KMeans

#1. 입력 데이터
# 각 행은 데이터, 각 열은 특성
# 각 데이터(행)는 [특성1, 특성2] 구조
# [수학점수, 영어점수], [구매횟수, 구매금액]
X = np.array([
    [1.0, 1.0],
    [1.5, 2.0],
    [2.0, 1.5],

    [8.0, 8.0],
    [9.0, 8.0],
    [8.0, 9.0]
])

#2. K-Means모델 생성
# 데이터를 2개의 군집으로 나누겠다는 뜻
model1 = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

#3. 비지도 학습: 정답이 없음.
# 데이터의 군집 + 중심점 찾기
model1.fit(X)

#4. 각 데이터가 어느 군집에 속했는지 확인
labels = model1.labels_

#5. 각 군집의 중심점 확인
centers = model1.cluster_centers_

print("군집 번호:", labels)
# [0 0 0 1 1 1]
print("군집 중심점:")
print(centers)
'''
[[1.5 1.5]
 [8.3 8.3]]
'''
