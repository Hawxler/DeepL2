# model.py
# ESP32 + MPU6050 동작 분류
# Keras Functional API / 1D CNN

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sensor_io import (
    SensorClient,
    CLASS_NAMES,
    SAMPLE_COUNT,
    to_features,
    countdown,
)


# =========================================================
# 1. 기본 설정
# =========================================================
SEED = 42
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
VALIDATION_RATIO = 0.2

# 실행하는 터미널 위치와 관계없이 model.py 위치를 기준으로 설정
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "gesture.keras"
PREPROCESS_PATH = ARTIFACT_DIR / "preprocess.npz"


# =========================================================
# 2. CSV 데이터 읽기
# =========================================================
def load_split(split):
    """
    예:
    data/train/triangle/practice01/파일.csv

    반환:
    X: (동작 수, SAMPLE_COUNT, 6)
    y: (동작 수,)
    """
    features = []
    labels = []

    for label_index, class_name in enumerate(CLASS_NAMES):
        folder = DATA_DIR / split / class_name

        # practice01 같은 하위 폴더까지 검색
        files = sorted(folder.rglob("*.csv"))

        if not files:
            raise ValueError(
                f"CSV 파일이 없습니다: {folder}\n"
                f"'{split}' 데이터의 '{class_name}' 동작을 수집하세요."
            )

        print(f"{split:5s} / {class_name:8s}: {len(files)}")

        for file in files:
            try:
                # 첫 행은 t_us,ax,ay,az,gx,gy,gz 헤더
                raw = np.loadtxt(
                    file,
                    delimiter=",",
                    skiprows=1,
                )

                # (100, 7) -> (100, 6)
                sample = to_features(raw)

            except ValueError as error:
                raise ValueError(
                    f"CSV 데이터 오류: {file}\n{error}"
                ) from error

            features.append(sample)
            labels.append(label_index)

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)

    return X, y


# =========================================================
# 3. 1D CNN 모델 생성
# =========================================================
def build_model():
    inputs = tf.keras.Input(
        shape=(SAMPLE_COUNT, 6),
        name="sensor_input",
    )

    # 입력: (100, 6)
    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv1",
    )(inputs)

    # (100, 32) -> (50, 32)
    x = tf.keras.layers.MaxPooling1D(
        pool_size=2,
        name="pool1",
    )(x)

    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv2",
    )(x)

    # (50, 64) -> (25, 64)
    x = tf.keras.layers.MaxPooling1D(
        pool_size=2,
        name="pool2",
    )(x)

    # (25, 64) -> (1600,)
    x = tf.keras.layers.Flatten(name="flatten")(x)

    x = tf.keras.layers.Dense(
        units=32,
        activation="relu",
        name="hidden",
    )(x)

    x = tf.keras.layers.Dropout(
        rate=0.3,
        name="dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        units=len(CLASS_NAMES),
        activation="softmax",
        name="gesture_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="gesture_1dcnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =========================================================
# 4. 학습
# =========================================================
def train():
    tf.keras.utils.set_random_seed(SEED)

    print("\n데이터 폴더:", DATA_DIR)

    # data/train에서 전체 데이터 읽기
    X_all, y_all = load_split("train")

    # 클래스 비율을 유지한 분할을 위해 최소 개수 확인
    counts = np.bincount(
        y_all,
        minlength=len(CLASS_NAMES),
    )

    if np.any(counts < 2):
        raise ValueError(
            "학습/검증 분할에는 클래스별 최소 2개 이상의 "
            "동작 파일이 필요합니다."
        )

    # 동작 하나(파일 하나)를 단위로 80:20 분할
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
        train_count = int(np.sum(y_train == index))
        val_count = int(np.sum(y_val == index))

        print(
            f"{name:8s}: "
            f"train={train_count}, val={val_count}"
        )

    # 학습 데이터만으로 센서 특성별 평균/표준편차 계산
    # 결과 크기: (1, 1, 6)
    mean = X_train.mean(
        axis=(0, 1),
        keepdims=True,
    )

    std = X_train.std(
        axis=(0, 1),
        keepdims=True,
    )

    # 표준편차가 0에 가까운 경우 나눗셈 오류 방지
    std = np.maximum(std, 1e-6)

    # 검증 데이터에도 학습 데이터의 기준 적용
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    model = build_model()
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
    )

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

    # 최상의 검증 손실을 기록한 가중치로 평가
    val_loss, val_accuracy = model.evaluate(
        X_val,
        y_val,
        verbose=0,
    )

    print(f"\n검증 손실: {val_loss:.4f}")
    print(f"검증 정확도: {val_accuracy:.2%}")

    # 모델과 전처리 정보 저장
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(MODEL_PATH)

    np.savez(
        PREPROCESS_PATH,
        mean=mean,
        std=std,
        class_names=np.asarray(CLASS_NAMES),
    )

    print("\n저장 완료")
    print("모델:", MODEL_PATH)
    print("전처리 정보:", PREPROCESS_PATH)


# =========================================================
# 5. 저장된 모델과 전처리 정보 불러오기
# =========================================================
def load_artifacts():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {MODEL_PATH}\n"
            "먼저 python model.py train 을 실행하세요."
        )

    if not PREPROCESS_PATH.is_file():
        raise FileNotFoundError(
            f"전처리 파일이 없습니다: {PREPROCESS_PATH}\n"
            "먼저 python model.py train 을 실행하세요."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with np.load(PREPROCESS_PATH, allow_pickle=False) as data:
        mean = data["mean"]
        std = data["std"]
        names = data["class_names"].tolist()

    if names != list(CLASS_NAMES):
        raise ValueError(
            "저장된 모델의 클래스 순서와 "
            "sensor_io.py의 CLASS_NAMES가 다릅니다."
        )

    if model.input_shape != (None, SAMPLE_COUNT, 6):
        raise ValueError(
            f"모델 입력 크기가 다릅니다: {model.input_shape}"
        )

    return model, mean, std, names


# =========================================================
# 6. 별도로 수집한 test 데이터로 평가
# =========================================================
def evaluate():
    model, mean, std, names = load_artifacts()

    # data/test/triangle, circle, square 필요
    X_test, y_test = load_split("test")

    # 학습할 때 저장한 평균/표준편차 사용
    X_test = (X_test - mean) / std

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    probabilities = model.predict(
        X_test,
        verbose=0,
    )

    # 각 동작에서 점수가 가장 큰 클래스 선택
    predicted = probabilities.argmax(axis=1)

    print(f"\n테스트 손실: {loss:.4f}")
    print(f"테스트 정확도: {accuracy:.2%}")

    label_indices = list(range(len(names)))

    print("\n혼동행렬: 행=실제 정답, 열=예측")
    print("클래스 순서:", names)

    print(
        confusion_matrix(
            y_test,
            predicted,
            labels=label_indices,
        )
    )

    print("\n클래스별 평가")
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


# =========================================================
# 7. ESP32에서 새 동작을 받아 PC에서 분류
# =========================================================
def predict(ip):
    model, mean, std, names = load_artifacts()

    print(f"\nESP32 연결: {ip}")
    client = SensorClient(ip)

    try:
        while True:
            command = input(
                "\nEnter: 동작 측정 / q: 종료 > "
            ).strip().lower()

            if command == "q":
                break

            countdown()

            try:
                raw = client.record()
                features = to_features(raw)

            except ValueError as error:
                print(f"다시 측정하세요: {error}")
                continue

            print("STOP.")

            # (100, 6) -> (1, 100, 6)
            X = features[np.newaxis, ...]

            # 학습할 때와 동일하게 표준화
            X = (X - mean) / std

            probabilities = model.predict(
                X,
                verbose=0,
            )[0]

            index = int(np.argmax(probabilities))

            print(f"\n예측 결과: {names[index]}")

            for name, score in zip(names, probabilities):
                print(f"  {name:8s}: {score:.3f}")

    finally:
        client.close()


# =========================================================
# 8. 실행 모드 선택
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="MPU6050 동작 데이터 학습·평가·추론"
    )

    parser.add_argument(
        "mode",
        choices=["train", "evaluate", "predict"],
        help="train: 학습 / evaluate: 평가 / predict: 새 동작 분류",
    )

    parser.add_argument(
        "--ip",
        help="ESP32의 IP 주소. predict 모드에서 필요",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "evaluate":
        evaluate()

    elif args.mode == "predict":
        if not args.ip:
            parser.error(
                "predict 모드에서는 --ip가 필요합니다."
            )

        predict(args.ip)


if __name__ == "__main__":
    main()