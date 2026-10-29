# model.py의 주석 해설 버전

# ============================================================
# model.py
#
# 목적:
#   ESP32 + MPU6050으로 수집한 동작 데이터를 사용하여
#   삼각형, 원, 네모를 구분하는 1D CNN 모델을 학습합니다.
#
# 실행 방법:
#   1. 학습
#      python model.py train
#
#   2. 별도의 테스트 데이터로 평가
#      python model.py evaluate
#
#   3. ESP32에서 새로운 동작을 받아 PC에서 분류
#      python model.py predict --ip 192.168.0.5
#
# 주의:
#   이 코드는 PC에서 학습하고 PC에서 추론하는 버전입니다.
#   ESP32는 센서 데이터를 측정하여 PC에 전달합니다.
#
# 필요한 파일:
#   model.py
#   sensor_io.py
#
# 데이터 폴더 예:
#   data/train/triangle/practice01/파일.csv
#   data/train/circle/practice01/파일.csv
#   data/train/square/practice01/파일.csv
#
# 테스트 데이터 폴더 예:
#   data/test/triangle/test01/파일.csv
#   data/test/circle/test01/파일.csv
#   data/test/square/test01/파일.csv
#
# 데이터 단위:
#   CSV 파일 하나 = 도형을 한 번 그린 동작 하나
#
#   CSV 원본 크기:
#       (100, 7)
#       100개 시점 × [t_us, ax, ay, az, gx, gy, gz]
#
#   모델 입력용 동작 하나의 크기:
#       (100, 6)
#       100개 시점 × [ax, ay, az, gx, gy, gz]
#
#   여러 동작을 모은 입력 크기:
#       (동작 수, 100, 6)
# ============================================================


# ------------------------------------------------------------
# 명령줄 인자를 해석하는 파이썬 표준 라이브러리
#
# 예:
#   python model.py predict --ip 192.168.0.5
#
# 여기서 predict와 --ip 값을 읽는 데 사용합니다.
# ------------------------------------------------------------
import argparse

# 운영체제에 맞게 파일과 폴더 경로를 다루는 클래스입니다.
# Windows의 "\"와 Linux의 "/"를 직접 구분할 필요가 줄어듭니다.
from pathlib import Path

# 배열 생성, 평균·표준편차 계산, 파일 저장 등에 사용합니다.
import numpy as np

# 신경망 모델 생성, 학습, 저장, 추론에 사용합니다.
import tensorflow as tf

# 전체 데이터를 학습용과 검증용으로 나누는 함수입니다.
from sklearn.model_selection import train_test_split

# classification_report:
#   클래스별 precision, recall, f1-score 등을 출력합니다.
#
# confusion_matrix:
#   어떤 클래스를 어떤 클래스로 잘못 예측했는지 집계합니다.
from sklearn.metrics import classification_report, confusion_matrix

# 같은 폴더의 sensor_io.py에서 필요한 항목을 가져옵니다.
#
# SensorClient:
#   ESP32와 연결하고 센서 데이터를 받는 클래스입니다.
#
# CLASS_NAMES:
#   클래스 이름과 순서를 정의한 목록입니다.
#   기존 코드에서는 ["triangle", "circle", "square"]입니다.
#
# SAMPLE_COUNT:
#   동작 하나에 포함된 측정 시점 수입니다. 현재는 100입니다.
#
# to_features:
#   원본 데이터의 품질을 확인하고 시간축을 보간한 뒤,
#   시간 열을 제외한 센서 특성 6개를 반환합니다.
#
# countdown:
#   동작 시작 전에 3, 2, 1 카운트다운을 출력합니다.
from sensor_io import (
    SensorClient,
    CLASS_NAMES,
    SAMPLE_COUNT,
    to_features,
    countdown,
)


# ============================================================
# 1. 기본 설정
# ============================================================

# 난수 시드:
#   같은 데이터로 다시 실행할 때 데이터 분할과 초기화 결과를
#   가능한 한 재현하기 위해 사용합니다.
#
#   단, 하드웨어와 연산 환경에 따라 학습 결과가
#   완전히 동일하지 않을 수도 있습니다.
SEED = 42

# 최대 학습 횟수입니다.
# epoch 1회는 전체 학습 데이터를 한 번 사용하는 과정입니다.
#
# EarlyStopping 조건이 충족되면 100회 전에 종료됩니다.
EPOCHS = 100

# 한 번의 가중치 업데이트에 사용하는 동작 수입니다.
#
# 예:
#   학습 동작이 80개이고 batch_size=16이면
#   한 epoch에 5개 미니배치를 처리합니다.
#
# 여기서 16은 센서 측정 시점 수가 아니라
# "도형을 그린 동작의 개수"입니다.
BATCH_SIZE = 16

# Adam 옵티마이저의 학습률입니다.
# 가중치와 편향을 업데이트하는 크기에 영향을 줍니다.
LEARNING_RATE = 0.001

# 전체 데이터 중 검증용으로 사용할 비율입니다.
# 0.2는 검증 20%, 학습 80%를 의미합니다.
VALIDATION_RATIO = 0.2

# __file__:
#   현재 실행 중인 파이썬 파일(model.py)의 경로입니다.
#
# resolve():
#   해당 경로를 절대 경로로 변환합니다.
#
# parent:
#   파일이 들어 있는 상위 폴더를 가져옵니다.
#
# 예:
#   D:/0_pr/deepL2/esp32_imu_tfLite/model.py
#       ↓
#   D:/0_pr/deepL2/esp32_imu_tfLite
#
# 이 설정 덕분에 어느 터미널 폴더에서 실행하더라도
# model.py가 있는 폴더를 기준으로 데이터를 찾습니다.
BASE_DIR = Path(__file__).resolve().parent

# Path 객체에서 / 연산자는 경로를 연결합니다.
#
# DATA_DIR:
#   model.py 폴더 아래의 data 폴더
#
# ARTIFACT_DIR:
#   학습된 모델과 전처리 정보를 저장할 artifacts 폴더
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

# 학습된 Keras 모델의 저장 경로입니다.
#
# .keras 파일에는 모델 구조와 학습된 가중치 등이 저장됩니다.
# 이 파일 자체가 ESP32에서 바로 실행되는 .tflite 파일은 아닙니다.
MODEL_PATH = ARTIFACT_DIR / "gesture.keras"

# 표준화에 사용한 평균, 표준편차, 클래스 순서의 저장 경로입니다.
#
# .npz는 여러 NumPy 배열을 이름과 함께 저장하는 파일 형식입니다.
PREPROCESS_PATH = ARTIFACT_DIR / "preprocess.npz"


# ============================================================
# 2. CSV 데이터 읽기
# ============================================================
def load_split(split):
    """
    지정한 데이터 폴더의 CSV 파일들을 읽습니다.

    매개변수
    ----------
    split : str
        "train" 또는 "test"

        "train"이면 data/train 아래를 읽습니다.
        "test"이면 data/test 아래를 읽습니다.

    반환값
    ----------
    X : numpy.ndarray
        크기: (동작 수, SAMPLE_COUNT, 6)
        자료형: float32

    y : numpy.ndarray
        크기: (동작 수,)
        자료형: int32

        클래스 순서가 다음과 같다면:
            triangle -> 0
            circle   -> 1
            square   -> 2
    """

    # 각 CSV에서 읽은 (100, 6) 배열을 차례대로 담습니다.
    features = []

    # 각 동작에 해당하는 정수 라벨을 담습니다.
    # features[i]와 labels[i]가 하나의 입력·정답 쌍입니다.
    labels = []

    # enumerate()는 목록의 인덱스와 값을 함께 제공합니다.
    #
    # 첫 반복:
    #   label_index = 0
    #   class_name = "triangle"
    #
    # 두 번째 반복:
    #   label_index = 1
    #   class_name = "circle"
    #
    # 세 번째 반복:
    #   label_index = 2
    #   class_name = "square"
    for label_index, class_name in enumerate(CLASS_NAMES):

        # 예:
        #   DATA_DIR / "train" / "triangle"
        #
        # 최종적으로 다음과 같은 절대 경로가 됩니다.
        #   D:/.../esp32_imu_tfLite/data/train/triangle
        folder = DATA_DIR / split / class_name

        # rglob("*.csv"):
        #   현재 폴더뿐 아니라 모든 하위 폴더에서도
        #   확장자가 .csv인 파일을 찾습니다.
        #
        # 따라서 practice01, practice02 같은 폴더 안의
        # CSV도 모두 포함됩니다.
        #
        # sorted():
        #   파일 경로를 정렬하여 읽는 순서를 일정하게 만듭니다.
        #   학습 순서 섞기는 나중에 별도로 처리합니다.
        files = sorted(folder.rglob("*.csv"))

        # 파일 목록이 비어 있으면 오류를 발생시킵니다.
        # 폴더가 없을 때도 검색 결과가 비므로 이 조건에 걸립니다.
        if not files:
            raise ValueError(
                f"CSV 파일이 없습니다: {folder}\n"
                f"'{split}' 데이터의 '{class_name}' 동작을 수집하세요."
            )

        # :5s와 :8s는 문자열 출력 폭을 지정합니다.
        # 클래스별 파일 수를 보기 좋게 맞춰 출력합니다.
        print(f"{split:5s} / {class_name:8s}: {len(files)}")

        # 해당 클래스의 CSV 파일을 하나씩 처리합니다.
        for file in files:
            try:
                # delimiter=",":
                #   쉼표를 기준으로 열을 나눕니다.
                #
                # skiprows=1:
                #   첫 번째 행의 열 이름을 건너뜁니다.
                #
                # CSV 첫 행:
                #   t_us,ax,ay,az,gx,gy,gz
                #
                # 예상 raw 크기:
                #   (100, 7)
                raw = np.loadtxt(
                    file,
                    delimiter=",",
                    skiprows=1,
                )

                # sensor_io.py에 정의된 함수입니다.
                #
                # 주요 처리:
                #   1. 배열 크기 확인
                #   2. NaN, 무한대 확인
                #   3. 측정 간격과 전체 시간 확인
                #   4. 균일한 시간 간격으로 보간
                #   5. 시간 열을 제외한 센서 6개 열 반환
                #
                # raw:
                #   (100, 7)
                #
                # sample:
                #   (100, 6)
                #
                # 이 단계에서는 아직 평균·표준편차를 이용한
                # 표준화를 수행하지 않습니다.
                sample = to_features(raw)

            except ValueError as error:
                # 어느 CSV에서 문제가 발생했는지 경로를 포함하여
                # 새로운 오류 메시지를 만듭니다.
                #
                # from error:
                #   원래 발생한 오류와의 연결을 보존합니다.
                raise ValueError(
                    f"CSV 데이터 오류: {file}\n{error}"
                ) from error

            # CSV 하나에서 만든 동작 배열을 추가합니다.
            features.append(sample)

            # 폴더 이름에 대응하는 정수 라벨을 추가합니다.
            #
            # 예:
            #   triangle 폴더의 모든 파일에는 0을 부여합니다.
            #
            # 라벨을 센서 값으로 추측하는 것이 아니라,
            # 사람이 수집 시 지정한 폴더 이름을 정답으로 사용합니다.
            labels.append(label_index)

    # np.stack():
    #   같은 크기의 배열들을 새로운 축으로 쌓습니다.
    #
    # features에 (100, 6) 배열이 100개 있다면:
    #   X.shape == (100, 100, 6)
    #
    # 첫 번째 100:
    #   동작 수
    #
    # 두 번째 100:
    #   동작 하나에 포함된 측정 시점 수
    #
    # 6:
    #   각 시점의 센서 특성 수
    #
    # float32:
    #   신경망 계산에 사용할 32비트 실수형입니다.
    X = np.stack(features).astype(np.float32)

    # 정수 라벨 목록을 NumPy 배열로 변환합니다.
    #
    # 예:
    #   [0, 0, ..., 1, 1, ..., 2, 2, ...]
    #
    # 원-핫 인코딩을 하지 않고 정수 라벨을 사용합니다.
    y = np.asarray(labels, dtype=np.int32)

    # 입력 배열과 정답 배열을 함께 반환합니다.
    return X, y


# ============================================================
# 3. Keras Functional API로 1D CNN 생성
# ============================================================
def build_model():
    """
    시간에 따른 센서 값의 패턴을 학습하는 1D CNN을 만듭니다.

    입력:
        (배치 크기, 100, 6)

    출력:
        (배치 크기, 3)

    출력 열 순서:
        triangle, circle, square
    """

    # shape에는 배치 크기를 제외한 동작 하나의 크기를 지정합니다.
    #
    # SAMPLE_COUNT = 100인 경우:
    #   shape=(100, 6)
    #
    # 실제 모델 입력의 전체 크기:
    #   (None, 100, 6)
    #
    # None은 배치 크기가 고정되어 있지 않다는 뜻입니다.
    # 학습 시에는 보통 16개, 단일 추론 시에는 1개를 넣습니다.
    inputs = tf.keras.Input(
        shape=(SAMPLE_COUNT, 6),
        name="sensor_input",
    )

    # 첫 번째 1차원 합성곱 층입니다.
    #
    # filters=32:
    #   서로 다른 패턴을 찾는 필터 32개를 학습합니다.
    #   출력 특성의 개수가 32개가 됩니다.
    #
    # kernel_size=5:
    #   연속된 5개 시점을 한 번에 살펴봅니다.
    #   각 필터는 해당 구간의 센서 특성 6개를 함께 사용합니다.
    #
    # padding="same":
    #   기본 stride=1에서 시간축 길이를 유지합니다.
    #
    # activation="relu":
    #   합성곱 결과에 ReLU를 적용합니다.
    #   ReLU(z) = max(0, z)
    #
    # (inputs):
    #   생성한 층에 inputs를 연결한다는 뜻입니다.
    #
    # 크기 변화:
    #   (배치, 100, 6) -> (배치, 100, 32)
    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv1",
    )(inputs)

    # 시간축에서 인접한 2개 값 중 최댓값을 선택합니다.
    # 각 특성 채널별로 독립적으로 적용됩니다.
    #
    # pool_size=2:
    #   한 번에 2개 시점을 묶습니다.
    #
    # strides를 지정하지 않으면 pool_size와 같은 값이 적용됩니다.
    # 따라서 여기서는 2칸씩 이동합니다.
    #
    # 크기 변화:
    #   (배치, 100, 32) -> (배치, 50, 32)
    x = tf.keras.layers.MaxPooling1D(
        pool_size=2,
        name="pool1",
    )(x)

    # 두 번째 합성곱 층입니다.
    #
    # 첫 번째 층에서 추출한 32개 특성을 입력으로 받아,
    # 더 복합적인 시간 패턴을 표현하는 64개 특성을 만듭니다.
    #
    # 여기서 kernel_size=5는 원본 센서의 5개 시점이 아니라
    # 이전 층 출력의 연속된 5개 위치를 의미합니다.
    #
    # 크기 변화:
    #   (배치, 50, 32) -> (배치, 50, 64)
    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv2",
    )(x)

    # 두 번째 풀링입니다.
    #
    # 시간축 길이는 절반으로 줄고 채널 수는 유지됩니다.
    #
    # 크기 변화:
    #   (배치, 50, 64) -> (배치, 25, 64)
    x = tf.keras.layers.MaxPooling1D(
        pool_size=2,
        name="pool2",
    )(x)

    # 각 동작의 시간축과 특성축을 하나의 벡터로 펼칩니다.
    # 배치 축은 유지합니다.
    #
    # 25 × 64 = 1600
    #
    # 크기 변화:
    #   (배치, 25, 64) -> (배치, 1600)
    #
    # Flatten 자체에는 학습할 가중치가 없습니다.
    x = tf.keras.layers.Flatten(
        name="flatten",
    )(x)

    # 완전연결 은닉층입니다.
    #
    # 앞에서 추출한 1600개 값을 조합하여
    # 분류에 사용할 32개 출력을 만듭니다.
    #
    # 기본적으로 가중치와 편향을 모두 학습합니다.
    #
    # 크기 변화:
    #   (배치, 1600) -> (배치, 32)
    x = tf.keras.layers.Dense(
        units=32,
        activation="relu",
        name="hidden",
    )(x)

    # 학습 중 입력 원소를 약 30% 확률로 0으로 만듭니다.
    #
    # 특정 은닉 출력에 과도하게 의존하는 것을 줄여
    # 과적합을 완화하려는 목적입니다.
    #
    # 학습 시 남은 값에는 1 / (1 - 0.3) 배율이 적용됩니다.
    #
    # evaluate()나 predict()에서는 Dropout이 비활성화됩니다.
    # 배열 크기는 바뀌지 않습니다.
    x = tf.keras.layers.Dropout(
        rate=0.3,
        name="dropout",
    )(x)

    # 최종 분류 출력층입니다.
    #
    # len(CLASS_NAMES):
    #   클래스 개수입니다. 현재는 3입니다.
    #
    # softmax:
    #   3개 출력을 0~1 범위이며 합이 1인 점수로 변환합니다.
    #
    # 예:
    #   [0.80, 0.15, 0.05]
    #
    # 클래스 순서가 triangle, circle, square이므로
    # 위 예에서는 triangle의 점수가 가장 큽니다.
    #
    # 점수 0.80이 실제 정답률 80%를 보장하는 것은 아닙니다.
    #
    # 크기 변화:
    #   (배치, 32) -> (배치, 3)
    outputs = tf.keras.layers.Dense(
        units=len(CLASS_NAMES),
        activation="softmax",
        name="gesture_output",
    )(x)

    # 입력부터 출력까지 연결한 층들을 하나의 모델로 묶습니다.
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="gesture_1dcnn",
    )

    # 학습 방법을 설정합니다.
    #
    # optimizer:
    #   손실을 줄이도록 가중치와 편향을 갱신하는 방법입니다.
    #
    # Adam:
    #   기울기의 이동평균 등을 활용하는 옵티마이저입니다.
    #
    # loss="sparse_categorical_crossentropy":
    #   정답이 0, 1, 2처럼 정수로 주어진 다중 분류에 사용합니다.
    #   따라서 원-핫 인코딩은 필요하지 않습니다.
    #
    # metrics=["accuracy"]:
    #   학습 과정을 관찰할 때 정확도도 함께 계산합니다.
    #   가중치 업데이트의 직접적인 기준은 loss입니다.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# 4. 데이터 분할, 표준화, 학습, 저장
# ============================================================
def train():

    # Python, NumPy, TensorFlow 난수 시드를 설정합니다.
    # 모델의 초기 가중치와 학습 중 난수 사용의 재현성을 높입니다.
    tf.keras.utils.set_random_seed(SEED)

    print("\n데이터 폴더:", DATA_DIR)

    # data/train 아래에 있는 모든 클래스의 CSV를 읽습니다.
    #
    # 이 시점의 "train"은 폴더 이름입니다.
    # 읽은 데이터는 아래에서 실제 학습용과 검증용으로 나눕니다.
    #
    # 현재 데이터가 총 100개라면:
    #   X_all.shape = (100, 100, 6)
    #   y_all.shape = (100,)
    X_all, y_all = load_split("train")

    # 각 정수 라벨이 몇 번 등장하는지 셉니다.
    #
    # 예:
    #   삼각형 40개, 원 30개, 네모 30개이면
    #   counts = [40, 30, 30]
    #
    # minlength:
    #   결과 배열에 최소한 클래스 개수만큼의 항목을 만듭니다.
    counts = np.bincount(
        y_all,
        minlength=len(CLASS_NAMES),
    )

    # 클래스 비율을 유지하는 stratify 분할에는
    # 각 클래스에 최소 2개 이상의 데이터가 필요합니다.
    #
    # 단, 이것만으로 모든 소규모 데이터의 분할이 보장되지는 않습니다.
    # 학습·검증 각각의 전체 개수도 클래스 수 이상이어야 합니다.
    # 현재 100개 데이터의 80:20 분할에는 해당 문제가 없습니다.
    if np.any(counts < 2):
        raise ValueError(
            "학습/검증 분할에는 클래스별 최소 2개 이상의 "
            "동작 파일이 필요합니다."
        )

    # 전체 데이터를 학습용과 검증용으로 나눕니다.
    #
    # test_size=0.2:
    #   여기서는 전체 데이터의 20%를 검증용으로 사용합니다.
    #   인자 이름이 test_size이지만, 반환된 데이터를
    #   X_val, y_val에 담아 검증용으로 사용하는 것입니다.
    #
    # random_state=42:
    #   같은 입력 데이터라면 같은 방식으로 분할합니다.
    #
    # stratify=y_all:
    #   학습용과 검증용에 클래스 비율을 가능한 한 유지합니다.
    #
    # 중요:
    #   (100, 6)인 동작 전체를 하나의 단위로 나눕니다.
    #   동작 내부의 시간 순서 100개를 따로 섞지 않습니다.
    #
    # 총 100개 동작인 경우:
    #   X_train: (80, 100, 6)
    #   X_val:   (20, 100, 6)
    #   y_train: (80,)
    #   y_val:   (20,)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=VALIDATION_RATIO,
        random_state=SEED,
        stratify=y_all,
    )

    print("\n분할 결과")
    print("X_train:", X_train.shape)
    print("X_val  :", X_val.shape)
    print("y_train:", y_train.shape)
    print("y_val  :", y_val.shape)

    print("\n클래스별 데이터 개수")

    for index, name in enumerate(CLASS_NAMES):

        # y_train == index:
        #   해당 클래스이면 True, 아니면 False인 배열을 만듭니다.
        #
        # np.sum():
        #   True를 1, False를 0으로 계산하여 해당 클래스 수를 셉니다.
        #
        # int():
        #   NumPy 정수를 파이썬 정수로 변환합니다.
        train_count = int(np.sum(y_train == index))
        val_count = int(np.sum(y_val == index))

        print(
            f"{name:8s}: "
            f"train={train_count}, val={val_count}"
        )

    # --------------------------------------------------------
    # 표준화 기준 계산
    #
    # X_train의 축:
    #   axis=0: 동작 축
    #   axis=1: 시간 축
    #   axis=2: 센서 특성 축
    #
    # axis=(0, 1):
    #   동작 축과 시간 축을 함께 집계합니다.
    #   센서 특성 축은 남겨 둡니다.
    #
    # 따라서 ax, ay, az, gx, gy, gz 각각의 평균을 계산합니다.
    # 서로 다른 센서 특성 6개를 하나의 평균으로 합치지 않습니다.
    #
    # keepdims=True:
    #   집계한 축을 없애지 않고 길이 1로 유지합니다.
    #
    # 입력 크기:
    #   (80, 100, 6)
    #
    # 평균 크기:
    #   (1, 1, 6)
    #
    # 검증 데이터의 정보가 학습 전처리 기준에 섞이지 않도록
    # 반드시 X_train만 사용합니다.
    # --------------------------------------------------------
    mean = X_train.mean(
        axis=(0, 1),
        keepdims=True,
    )

    # 각 센서 특성의 표준편차를 계산합니다.
    #
    # 평균과 동일하게:
    #   std.shape == (1, 1, 6)
    std = X_train.std(
        axis=(0, 1),
        keepdims=True,
    )

    # 특성 값이 거의 일정하면 표준편차가 0에 가까울 수 있습니다.
    # 0으로 나누는 문제와 지나치게 큰 배율을 방지하기 위해
    # 표준편차의 최솟값을 0.000001로 제한합니다.
    std = np.maximum(std, 1e-6)

    # 표준화 공식:
    #
    #   표준화 값 = (원래 값 - 학습 평균) / 학습 표준편차
    #
    # 브로드캐스팅:
    #   (1, 1, 6)의 평균과 표준편차가
    #   모든 동작과 모든 시점에 특성별로 적용됩니다.
    #
    # 배열 크기는 바뀌지 않습니다.
    #
    # 검증 데이터에는 검증 데이터 자체의 평균·표준편차가 아니라
    # 학습 데이터에서 구한 mean과 std를 적용합니다.
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # 새로운 모델을 만듭니다.
    #
    # train 명령을 다시 실행하면 기존 모델을 이어서 학습하는 것이
    # 아니라 새 초기 가중치에서 학습을 시작합니다.
    #
    # 새 CSV를 추가했다면 기존 CSV와 함께 읽어 다시 학습합니다.
    model = build_model()

    # 층 이름, 출력 크기, 파라미터 수를 출력합니다.
    model.summary()

    # 검증 손실을 기준으로 학습을 일찍 멈추는 콜백입니다.
    #
    # monitor="val_loss":
    #   epoch가 끝날 때 검증 데이터의 손실을 확인합니다.
    #
    # patience=12:
    #   검증 손실이 개선되지 않는 epoch가 12회 연속 이어지면
    #   학습을 중단합니다.
    #
    # restore_best_weights=True:
    #   마지막 epoch의 가중치가 아니라,
    #   가장 낮은 검증 손실을 기록한 가중치로 복원합니다.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
    )

    # 실제 학습을 수행합니다.
    #
    # X_train:
    #   입력 센서 데이터
    #
    # y_train:
    #   각 동작의 정답 라벨
    #
    # validation_data:
    #   epoch가 끝날 때 성능을 확인할 데이터입니다.
    #   검증 데이터로 역전파하여 가중치를 갱신하지는 않습니다.
    #   다만 조기 종료와 최적 가중치 선택에 사용됩니다.
    #
    # epochs:
    #   최대 반복 횟수
    #
    # batch_size:
    #   가중치 업데이트 한 번에 사용할 동작 수
    #
    # shuffle=True:
    #   매 epoch마다 학습 동작들의 순서를 섞습니다.
    #   동작 내부의 시간 순서는 유지합니다.
    #
    # callbacks:
    #   학습 도중 실행할 추가 동작 목록
    #
    # verbose=2:
    #   epoch마다 결과를 한 줄씩 출력합니다.
    #
    # 내부적으로 각 학습 배치에서 다음 과정을 수행합니다.
    #   순전파 -> 손실 계산 -> 역전파 -> 가중치·편향 업데이트
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[early_stop],
        verbose=2,
    )

    # 복원된 최적 가중치로 검증 성능을 계산합니다.
    #
    # evaluate()는 가중치를 갱신하지 않습니다.
    # Dropout도 비활성화됩니다.
    #
    # compile()에서 loss와 accuracy를 지정했으므로
    # 여기서는 손실과 정확도 두 값을 받습니다.
    val_loss, val_accuracy = model.evaluate(
        X_val,
        y_val,
        verbose=0,
    )

    # :.4f:
    #   소수점 아래 4자리로 출력합니다.
    #
    # :.2%:
    #   0.85와 같은 값을 85.00% 형식으로 출력합니다.
    print(f"\n검증 손실: {val_loss:.4f}")
    print(f"검증 정확도: {val_accuracy:.2%}")

    # 저장 폴더가 없으면 만듭니다.
    #
    # parents=True:
    #   필요한 상위 폴더도 함께 생성합니다.
    #
    # exist_ok=True:
    #   이미 폴더가 있어도 오류를 내지 않습니다.
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 모델 구조와 학습된 가중치 등을 저장합니다.
    # 같은 경로에 기존 파일이 있으면 갱신됩니다.
    model.save(MODEL_PATH)

    # 여러 배열을 하나의 .npz 파일에 저장합니다.
    #
    # 저장할 배열 이름:
    #   mean
    #   std
    #   class_names
    #
    # 추론 시 이 값을 그대로 불러와 사용해야
    # 학습 때와 동일한 전처리를 적용할 수 있습니다.
    #
    # mean과 std는 모델 파일 안에 자동 포함되는 것이 아니라
    # 이 코드에서는 별도 파일로 관리합니다.
    np.savez(
        PREPROCESS_PATH,
        mean=mean,
        std=std,
        class_names=np.asarray(CLASS_NAMES),
    )

    print("\n저장 완료")
    print("모델:", MODEL_PATH)
    print("전처리 정보:", PREPROCESS_PATH)


# ============================================================
# 5. 저장된 모델과 전처리 정보 불러오기
# ============================================================
def load_artifacts():
    """
    학습 결과를 읽어 반환합니다.

    반환:
        model: 학습된 Keras 모델
        mean:  학습 데이터로 계산한 평균, (1, 1, 6)
        std:   학습 데이터로 계산한 표준편차, (1, 1, 6)
        names: 클래스 이름 목록
    """

    # 모델 파일이 없으면 학습부터 실행하도록 안내합니다.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {MODEL_PATH}\n"
            "먼저 python model.py train 을 실행하세요."
        )

    # 전처리 값 없이 모델만 사용하면
    # 학습 당시와 다른 입력 값이 들어갈 수 있으므로 확인합니다.
    if not PREPROCESS_PATH.is_file():
        raise FileNotFoundError(
            f"전처리 파일이 없습니다: {PREPROCESS_PATH}\n"
            "먼저 python model.py train 을 실행하세요."
        )

    # 저장한 모델 구조와 가중치 등을 복원합니다.
    model = tf.keras.models.load_model(MODEL_PATH)

    # np.load():
    #   .npz 파일에 저장된 배열들을 읽습니다.
    #
    # allow_pickle=False:
    #   파이썬 객체 역직렬화를 허용하지 않습니다.
    #   여기서는 숫자와 문자열 배열만 저장했으므로 필요하지 않습니다.
    #
    # with:
    #   블록이 끝나면 열린 파일 자원을 정리합니다.
    with np.load(PREPROCESS_PATH, allow_pickle=False) as data:
        mean = data["mean"]
        std = data["std"]

        # NumPy 문자열 배열을 파이썬 목록으로 변환합니다.
        names = data["class_names"].tolist()

    # 클래스 순서가 달라지면 출력 점수를 잘못 해석하게 됩니다.
    #
    # 예:
    #   모델은 0번을 triangle로 학습했는데
    #   현재 코드가 0번을 circle로 해석하면 결과가 틀어집니다.
    if names != list(CLASS_NAMES):
        raise ValueError(
            "저장된 모델의 클래스 순서와 "
            "sensor_io.py의 CLASS_NAMES가 다릅니다."
        )

    # 입력 크기가 현재 센서 처리 방식과 같은지 확인합니다.
    #
    # 예상:
    #   (None, 100, 6)
    #
    # None:
    #   가변적인 배치 크기
    if model.input_shape != (None, SAMPLE_COUNT, 6):
        raise ValueError(
            f"모델 입력 크기가 다릅니다: {model.input_shape}"
        )

    return model, mean, std, names


# ============================================================
# 6. 별도로 수집한 test 데이터로 평가
# ============================================================
def evaluate():
    """
    학습에 사용하지 않은 data/test 데이터로 성능을 평가합니다.

    train()에서 나눈 검증 데이터와는 별도입니다.
    """

    # 저장된 모델과 학습 당시 전처리 기준을 가져옵니다.
    model, mean, std, names = load_artifacts()

    # 다음 폴더의 CSV 파일을 읽습니다.
    #
    # data/test/triangle/...
    # data/test/circle/...
    # data/test/square/...
    X_test, y_test = load_split("test")

    # 테스트 데이터 자체에서 평균·표준편차를 다시 계산하지 않습니다.
    # 반드시 학습 때 저장한 값을 사용합니다.
    X_test = (X_test - mean) / std

    # 테스트 전체의 손실과 정확도를 계산합니다.
    # 모델의 가중치는 변하지 않습니다.
    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    # 각 테스트 동작의 클래스별 점수를 계산합니다.
    #
    # 테스트 동작이 60개라면:
    #   probabilities.shape == (60, 3)
    #
    # 예:
    #   [
    #       [0.80, 0.15, 0.05],
    #       [0.10, 0.75, 0.15],
    #       ...
    #   ]
    probabilities = model.predict(
        X_test,
        verbose=0,
    )

    # axis=1:
    #   각 행 안에서 클래스 열 3개를 비교합니다.
    #
    # argmax:
    #   가장 큰 값 자체가 아니라 그 값의 인덱스를 반환합니다.
    #
    # 예:
    #   [0.10, 0.75, 0.15] -> 1 -> circle
    #
    # 결과 크기:
    #   (테스트 동작 수,)
    predicted = probabilities.argmax(axis=1)

    print(f"\n테스트 손실: {loss:.4f}")
    print(f"테스트 정확도: {accuracy:.2%}")

    # 클래스 인덱스 목록입니다.
    # 클래스가 3개이면 [0, 1, 2]입니다.
    label_indices = list(range(len(names)))

    print("\n혼동행렬: 행=실제 정답, 열=예측")
    print("클래스 순서:", names)

    # 혼동행렬의 예:
    #
    #              예측
    #           세모 원 네모
    # 실제 세모  18  1  1
    # 실제 원     0 19  1
    # 실제 네모   2  1 17
    #
    # 대각선:
    #   올바르게 분류한 개수
    #
    # 대각선 이외:
    #   다른 클래스로 잘못 분류한 개수
    #
    # labels:
    #   행과 열의 클래스 순서를 명시합니다.
    print(
        confusion_matrix(
            y_test,
            predicted,
            labels=label_indices,
        )
    )

    print("\n클래스별 평가")

    # precision:
    #   해당 클래스라고 예측한 것 중 실제로 맞은 비율
    #
    # recall:
    #   실제 해당 클래스인 것 중 올바르게 찾아낸 비율
    #
    # f1-score:
    #   precision과 recall의 조화평균
    #
    # support:
    #   실제 해당 클래스의 테스트 데이터 개수
    #
    # target_names:
    #   숫자 라벨 대신 표시할 클래스 이름
    #
    # digits=3:
    #   소수점 아래 3자리 표시
    #
    # zero_division=0:
    #   분모가 0이어서 계산할 수 없는 지표를 0으로 처리
    print(
        classification_report(
            y_test,
            predicted,
            labels=label_indices,
            target_names=names,
            digits=3,
            zero_division=0,
        )
    )


# ============================================================
# 7. ESP32에서 새로운 동작을 받아 PC에서 추론
# ============================================================
def predict(ip):
    """
    ESP32에 연결하여 새 동작을 반복 측정하고 분류합니다.

    매개변수:
        ip: ESP32의 IP 주소 문자열
    """

    # 반복문 전에 모델을 한 번만 읽습니다.
    # 동작을 측정할 때마다 모델을 다시 읽을 필요가 없습니다.
    model, mean, std, names = load_artifacts()

    print(f"\nESP32 연결: {ip}")

    # sensor_io.py의 SensorClient를 생성합니다.
    # 내부에서 ESP32의 TCP 서버에 연결합니다.
    client = SensorClient(ip)

    # try/finally:
    #   정상 종료하거나 도중에 오류가 나더라도
    #   finally에서 연결을 닫도록 구성합니다.
    try:
        while True:

            # input():
            #   사용자가 Enter를 누를 때까지 기다립니다.
            #
            # strip():
            #   입력 앞뒤의 공백을 제거합니다.
            #
            # lower():
            #   Q를 입력해도 q로 처리하도록 소문자로 변환합니다.
            command = input(
                "\nEnter: 동작 측정 / q: 종료 > "
            ).strip().lower()

            # q이면 반복문을 종료합니다.
            if command == "q":
                break

            # 3, 2, 1 카운트다운과 GO 메시지를 출력합니다.
            countdown()

            try:
                # ESP32에 측정을 요청하고 원본 데이터를 받습니다.
                #
                # 예상 크기:
                #   raw.shape == (100, 7)
                #
                # 기존 SensorClient.record() 내부에서도
                # to_features(raw)를 호출하여 품질을 확인하지만,
                # 반환하는 값은 원본 raw입니다.
                raw = client.record()

                # 원본을 모델 입력용 센서 배열로 변환합니다.
                #
                # 예상 크기:
                #   features.shape == (100, 6)
                features = to_features(raw)

            except ValueError as error:
                # 측정 간격이나 데이터 크기 등에 문제가 있으면
                # 해당 동작은 추론하지 않고 다시 입력을 기다립니다.
                print(f"다시 측정하세요: {error}")
                continue

            print("STOP.")

            # 모델은 동작 하나를 넣어도 배치 축을 필요로 합니다.
            #
            # features:
            #   (100, 6)
            #
            # features[np.newaxis, ...]:
            #   (1, 100, 6)
            #
            # np.newaxis:
            #   길이 1인 새 축을 맨 앞에 추가합니다.
            #
            # ...:
            #   나머지 모든 축을 그대로 유지합니다.
            X = features[np.newaxis, ...]

            # 학습 데이터에서 계산한 기준으로 표준화합니다.
            # 새 동작 자체의 평균·표준편차를 사용하지 않습니다.
            X = (X - mean) / std

            # 모델의 출력 크기는 (1, 3)입니다.
            #
            # [0]:
            #   배치에서 첫 번째이자 유일한 동작의 결과를 꺼냅니다.
            #
            # 최종 probabilities 크기:
            #   (3,)
            #
            # 이 [0]은 첫 번째 클래스만 고르는 것이 아닙니다.
            # 동작 하나에 대한 클래스 점수 3개 전체를 꺼냅니다.
            probabilities = model.predict(
                X,
                verbose=0,
            )[0]

            # 길이 3의 배열에서 가장 큰 점수의 위치를 찾습니다.
            #
            # 예:
            #   probabilities = [0.1, 0.2, 0.7]
            #   index = 2
            #
            # int():
            #   NumPy 정수를 파이썬 정수로 변환합니다.
            index = int(np.argmax(probabilities))

            # 인덱스에 해당하는 클래스 이름을 출력합니다.
            print(f"\n예측 결과: {names[index]}")

            # zip():
            #   클래스 이름과 점수를 같은 순서로 하나씩 묶습니다.
            #
            # 예:
            #   ("triangle", 0.1)
            #   ("circle", 0.2)
            #   ("square", 0.7)
            #
            # :.3f:
            #   점수를 소수점 아래 3자리로 표시합니다.
            for name, score in zip(names, probabilities):
                print(f"  {name:8s}: {score:.3f}")

    finally:
        # 파일 형태의 수신 스트림과 TCP 소켓을 닫습니다.
        client.close()


# ============================================================
# 8. 명령줄 인자를 해석하여 실행 모드 선택
# ============================================================
def main():

    # 명령줄 사용법과 인자를 관리할 파서를 생성합니다.
    #
    # 다음 명령으로 도움말을 확인할 수 있습니다.
    #   python model.py --help
    parser = argparse.ArgumentParser(
        description="MPU6050 동작 데이터 학습·평가·추론"
    )

    # mode는 위치 인자입니다.
    #
    # 예:
    #   python model.py train
    #
    # choices:
    #   세 가지 문자열 중 하나만 허용합니다.
    #
    # mode를 생략하면 argparse가 사용법과 오류를 출력합니다.
    parser.add_argument(
        "mode",
        choices=["train", "evaluate", "predict"],
        help="train: 학습 / evaluate: 평가 / predict: 새 동작 분류",
    )

    # --ip는 선택 인자입니다.
    #
    # train/evaluate:
    #   저장된 CSV를 사용하므로 필요 없습니다.
    #
    # predict:
    #   ESP32에 연결해야 하므로 필요합니다.
    parser.add_argument(
        "--ip",
        help="ESP32의 IP 주소. predict 모드에서 필요",
    )

    # 사용자가 입력한 명령줄을 실제로 해석합니다.
    #
    # 예:
    #   python model.py predict --ip 192.168.0.5
    #
    # 결과:
    #   args.mode == "predict"
    #   args.ip == "192.168.0.5"
    args = parser.parse_args()

    # 지정한 모드에 따라 해당 함수를 실행합니다.
    if args.mode == "train":
        train()

    elif args.mode == "evaluate":
        evaluate()

    elif args.mode == "predict":

        # predict인데 IP 주소가 없으면 실행을 중단하고 안내합니다.
        if not args.ip:
            parser.error(
                "predict 모드에서는 --ip가 필요합니다."
            )

        predict(args.ip)


# ============================================================
# 9. 프로그램 시작점
# ============================================================

# 이 파일을 직접 실행하면 __name__은 "__main__"이 됩니다.
# 따라서 main()을 호출합니다.
#
# 다른 파일에서 import model로 가져오면 __name__은 "model"이므로
# main()은 자동 실행되지 않습니다.
# 함수와 설정을 가져와 재사용할 수 있습니다.
if __name__ == "__main__":
    main()