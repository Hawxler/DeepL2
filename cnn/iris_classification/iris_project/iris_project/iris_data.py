"""네 모델에서 공유하는 데이터 준비와 새 입력의 표준화."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class IrisData:
    DEFAULT_SOURCE = "https://raw.githubusercontent.com/hawklux/DeepLearn/main/0_Data/bong/iris.csv"
    feature_columns = ['꽃받침길이', '꽃받침폭', '꽃잎길이', '꽃잎폭']
    class_names = np.array(['setosa', 'versicolor', 'virginica'])

    def __init__(self, source=DEFAULT_SOURCE, test_size=0.2, seed=42):
        data = pd.read_csv(source)
        X = data[self.feature_columns].to_numpy(dtype=np.float32)
        mapping = {name: i for i, name in enumerate(self.class_names)}
        labels = data['품종'].map(mapping)
        if labels.isna().any():
            raise ValueError('알 수 없거나 누락된 품종 값이 있습니다.')
        if not np.isfinite(X).all():
            raise ValueError('입력에 결측값 또는 무한대가 있습니다.')
        y = labels.to_numpy(dtype=np.int64)
        X_train, X_test, self.y_train_index, self.y_test_index = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        self.scaler = StandardScaler()
        # 테스트 데이터의 정보가 학습에 들어가지 않도록 학습 데이터에만 fit.
        self.X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        self.X_test = self.scaler.transform(X_test).astype(np.float32)
        self.y_train = np.eye(3, dtype=np.float32)[self.y_train_index]
        self.y_test = np.eye(3, dtype=np.float32)[self.y_test_index]

    def transform(self, rows):
        """원래 단위의 (샘플 수, 4) 입력을 학습 때의 기준으로 변환."""
        X = np.asarray(rows, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != 4 or not np.isfinite(X).all():
            raise ValueError('새 입력은 유한한 숫자로 된 (샘플 수, 4) 배열이어야 합니다.')
        return self.scaler.transform(X).astype(np.float32)
