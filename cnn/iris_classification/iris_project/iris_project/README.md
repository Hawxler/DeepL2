# Iris 품종 분류: 네 가지 버전

## 1. 파일 구성

| 파일 | 역할 |
| --- | --- |
| iris_data.py | IrisData 클래스: CSV 읽기, 입력/정답 분리, 학습/테스트 분할, 표준화, 원-핫 인코딩 |
| keras_models.py | Keras Sequential / Functional API 모델 생성 함수 |
| torch_models.py | PyTorch Sequential 생성 함수 / forward() 직접 작성 클래스 |
| keras_runner.py | Keras 학습, 평가, 예측 함수 |
| torch_runner.py | PyTorch 학습, 평가, 예측 함수 |
| main.py | 모델 선택 및 전체 실행 |

파일을 합치지 않습니다. 여섯 Python 파일을 같은 폴더에 두고 main.py만 실행합니다.
IrisData는 학습 때 사용한 scaler를 보관하므로 클래스로 구성했습니다.
학습·평가·예측은 모델과 데이터를 인자로 받는 함수로 구성했습니다.

## 2. 실행 매뉴얼 (Windows PowerShell)

1. ZIP을 풀고 여섯 Python 파일이 들어 있는 iris_project 폴더를 원하는 위치에 놓습니다.
   예: D:\0_pr\deepL2\iris_project
   예상 결과: 이 폴더 안에 main.py와 다른 다섯 Python 파일이 보입니다.

2. 터미널에서 해당 폴더로 이동합니다.

   ```powershell
   cd D:\0_pr\deepL2\iris_project
   ```

3. 기존 가상환경의 Python으로 필요한 패키지를 설치합니다. 이미 설치되어 있으면 생략합니다.
   아래 가상환경 경로는 기존 프로젝트에 맞춘 예시입니다.

   ```powershell
   & D:\0_pr\deepL2\myenv\Scripts\python.exe -m pip install numpy pandas scikit-learn
   ```

   Keras 버전을 쓸 때:

   ```powershell
   & D:\0_pr\deepL2\myenv\Scripts\python.exe -m pip install tensorflow
   ```

   PyTorch 버전을 쓸 때:

   ```powershell
   & D:\0_pr\deepL2\myenv\Scripts\python.exe -m pip install torch
   ```

4. 원하는 버전의 명령 하나를 실행합니다.

   ```powershell
   # Keras Sequential
   & D:\0_pr\deepL2\myenv\Scripts\python.exe main.py --model keras-sequential

   # Keras Functional API
   & D:\0_pr\deepL2\myenv\Scripts\python.exe main.py --model keras-functional

   # PyTorch Sequential
   & D:\0_pr\deepL2\myenv\Scripts\python.exe main.py --model torch-sequential

   # PyTorch forward() 직접 작성
   & D:\0_pr\deepL2\myenv\Scripts\python.exe main.py --model torch-forward
   ```

   가상환경이 활성화되어 있으면 긴 Python 경로 대신 `python`을 사용해도 됩니다.
   VS Code 실행 버튼으로 main.py를 실행하면 기본값인 Keras Sequential이 선택됩니다.

5. 출력 결과를 확인합니다.

   - 학습 입력/정답: `(120, 4) (120, 3)`
   - 테스트 입력/정답: `(30, 4) (30, 3)`
   - 모델 구조 및 학습 중 loss와 accuracy
   - 학습 후 테스트 loss, accuracy
   - 예시 입력 3개의 예측 품종과 클래스별 확률

   loss는 예측 오차로, 대체로 감소하면 학습이 진행되고 있음을 뜻합니다.
   매 epoch마다 반드시 감소하지는 않습니다. accuracy는 정답률입니다.
   예측 예시는 사용법을 보여주기 위한 입력이며 독립적인 성능 검증 자료가 아닙니다.
   정확한 수치는 초기화, 실행 환경, 모델에 따라 달라집니다.

## 3. 설정 변경

```powershell
python main.py --model torch-forward --epochs 100 --batch-size 16 --lr 0.001
python main.py --model keras-functional --csv "D:\data\iris.csv"
```

| 인자 | 기본값 | 의미 |
| --- | --- | --- |
| --model | keras-sequential | 사용할 모델 버전 |
| --epochs | 200 | 학습 데이터 전체를 반복 학습하는 횟수 |
| --batch-size | 16 | 한 번의 가중치 갱신에 사용하는 샘플 수 |
| --lr | 0.001 | Adam 학습률 |
| --csv | GitHub URL | CSV의 URL 또는 로컬 파일 경로 |

기본 데이터 URL:
https://raw.githubusercontent.com/hawklux/DeepLearn/main/0_Data/bong/iris.csv

CSV 헤더는 `꽃받침길이,꽃받침폭,꽃잎길이,꽃잎폭,품종`이어야 합니다.
입력에는 앞의 네 열만 사용합니다. 품종은 setosa / versicolor / virginica입니다.
PyTorch BatchNorm은 학습 배치가 1개인 경우 오류가 나므로 기본 배치 크기 16을 권장합니다.

## 4. 중요한 함수와 인자

### 공통 데이터 준비
- `train_test_split(..., stratify=y)`: 품종 비율을 유지하며 학습 80%, 테스트 20%로 분리합니다.
- `random_state=42`: 같은 데이터에 대해 같은 분할을 재현합니다.
- `scaler.fit_transform()`: 학습 데이터로 표준화 기준을 계산하고 변환합니다.
- `scaler.transform()`: 테스트/새 입력에 학습 때 정한 기준을 적용합니다.
- `np.eye(3)[y]`: 정수 정답을 길이 3의 원-핫 벡터로 변환합니다.
- `data.transform(rows)`: 새 입력을 표준화합니다. 입력 순서는 반드시 네 특성 순서와 같아야 합니다.

### Keras Sequential
- `Sequential([...])`: 나열한 순서대로 층을 연결합니다.
- `Input(shape=(4,))`: 샘플 하나의 입력 특성이 4개입니다. 배치 크기는 포함하지 않습니다.
- `Dense(16)`: 출력 특성이 16개인 완전연결층입니다. 샘플 수는 바뀌지 않습니다.

### Keras Functional API
- `Dense(16)(inputs)`: 층을 생성하고 지정한 입력에 연결합니다.
- `Model(inputs, outputs)`: 입력부터 출력까지의 연결을 하나의 모델로 묶습니다.
- 이 예제의 층 구성은 Sequential과 동일하며, 연결을 작성하는 방식이 다릅니다.

### Keras 공통 학습
- `Adam(learning_rate=...)`: 학습률을 정합니다.
- `CategoricalCrossentropy()`: softmax 출력과 원-핫 정답을 비교합니다.
- `fit()`: 배치 처리, 역전파, 가중치 갱신을 수행합니다.
- `evaluate()` / `predict()`: 평가 / 추론을 수행하며 Dropout과 BatchNorm을 추론 방식으로 처리합니다.

### PyTorch Sequential
- `nn.Sequential(...)`: 층을 순서대로 실행하는 모델입니다.
- `nn.Linear(4, 16)`: 입력 특성 4개를 출력 특성 16개로 바꿉니다. 활성화함수가 아닙니다.

### PyTorch forward() 직접 작성
- `nn.Module`: 사용자 정의 모델의 기본 클래스입니다.
- `__init__()`: 학습할 층을 생성하고 등록합니다.
- `forward()`: 입력이 각 층을 통과하는 과정을 작성합니다.
- `model(x)`: 등록된 forward()를 호출합니다. Keras Functional API와 같은 이름의 API는 아닙니다.

### PyTorch 공통 학습
- `TensorDataset` / `DataLoader`: 입력·정답을 묶고 배치 단위로 제공합니다.
- `CrossEntropyLoss()`: logits와 정답을 비교합니다. 여기서는 float32 원-핫 정답을 사용합니다.
- `zero_grad()`: 이전 배치의 기울기를 지웁니다.
- `backward()`: 현재 손실의 기울기를 계산합니다.
- `step()`: 가중치를 갱신합니다.
- `train()` / `eval()`: Dropout·BatchNorm의 학습/평가 동작을 선택합니다.
- `no_grad()`: 추론 시 기울기 기록을 끕니다. eval()과 역할이 다릅니다.
- `softmax(..., dim=1)`: 예측 단계에서 각 샘플의 logits를 품종 확률로 바꿉니다.

## 5. 모델 구성과 주의점

네 모델의 기본 구조는 입력 특성 4개 → 은닉층 뉴런 16개 → 은닉층 뉴런 8개 → 출력 3개입니다.
은닉층은 2개입니다. 입력이 `(배치 크기, 4)`이면 첫 Linear/Dense 출력은 `(배치 크기, 16)`입니다.
Batch Normalization, ReLU, Dropout(0.2)을 포함합니다.
Dropout은 학습 중 일부 활성값을 확률 0.2로 0으로 만듭니다.
Pooling과 Flatten은 이 2차원 표 데이터에는 필요하지 않아 넣지 않았습니다.
Keras와 PyTorch는 초기화 및 BatchNorm 기본 설정 등이 달라 학습 결과가 완전히 같지는 않습니다.

테스트 세트는 학습 후 평가에만 사용합니다. 반복적인 튜닝에는 별도 검증 세트를 두는 것이 좋습니다.
현재 프로그램은 실행할 때마다 새로 학습하고 같은 실행 안에서 추론합니다.
나중에 학습 없이 추론하려면 모델과 scaler, 특성 순서, 클래스 순서를 함께 저장/복원해야 합니다.
