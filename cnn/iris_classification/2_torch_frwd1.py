# PyTorch forward() 직접 작성 코드

######################################
# 1. 데이터 준비(0_dataPrep1.py의 내용)
######################################
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

######################################
# 2. 모델 구축(PyTorch Sequential API)
# python 3.10 가상환경에 pytorch 설치
#  .\myenv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
######################################
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

class IrisModel1(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden1 = nn.Linear(4, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.2)

        self.hidden2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)

        self.relu = nn.ReLU()
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        x = self.hidden(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.relu(x)

        logits = self.output(x)
        return logits

model = IrisModel1()
print(model)

#################################################
# 3. 학습
#################################################
# 3.1 Numpy -> Tensor
x_train_tensor = torch.from_numpy(X_train) # (120, 4): [5.1, 3.5, 1.4, 0.2]
y_train_tensor = torch.from_numpy(y_train) #원-핫 (120, 3): [1.0, 0.0, 0.0]

x_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)

# 3.2 배치 준비
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
)
# DataLoader(): 데이터셋을 shuffle하고 모델이 학습하기 좋은 크기로 batch로 묶어서 for문으로 한 묶음씩 계속 공급해줌.
# print(train_dataset[0]) (tensor([-1.7216, -0.3248, -1.3470, -1.3202]), tensor([1., 0., 0.]))

# 3.3 손실함수와 옵티마이저 설정
# CrossEntropyloss는 로짓과 실수 원-핫 정답도 받을 수 있음.
loss_fn = nn.CrossEntropyLoss() # logit을 받아 로그 확률 도출 = 출력 레이어에서 suftmax 없어도 argmax로 정답 도출
optimizer = torch.optim.Adam(model.parameters(), li=0.001)

# 3.4 학습
for epoch in range(200):
    model.train()
    total_loss = 0.0
    correct = 0
    sample_count = 0

    # 배치(크기 최대 16)별 계산: 손실-기울기 계산 > 파라미터 업데이트
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad() # 


#################################################
# 4. 평가 > 추론: Sequential/Functinal 동일
#################################################