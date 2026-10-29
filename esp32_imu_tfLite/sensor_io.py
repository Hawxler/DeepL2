import socket
import time

import numpy as np

SAMPLE_COUNT = 100
SAMPLE_HZ = 50

FEATURE_NAMES = ["ax", "ay", "az", "gx", "gy", "gz"]
CLASS_NAMES = ["triangle", "circle", "square"]

def to_features(raw):
    """원본 (100, 7) 데이터를 모델 입력 (100, 6)으로 변환"""
    raw = np.asarray(raw, dtype=np.float64)

    if raw.shape !=(SAMPLE_COUNT, 7):
        raise ValueError(f"잘못된 데이터 크기: {raw.shape}")
    
    if not np.isfinite(raw).all():
        raise ValueError("데이터에 NaN 또는 무한대가 있습니다.")
    
    # 측정 시간을 마이크로초에서 초로 변환
    # 첫 번째 측정 시점을 0초로 설정
    times = (raw[:, 0] - raw[0, 0]) / 1_000_000.0
    intervals = np.diff(times)

    # 측정 간격 검사: 목표 20ms, 허용범위 10~30ms
    if np.any(intervals < 0.010) or np.any(intervals > 0.030):
        raise ValueError("측정 간격이 불안정함. 다시 측정하세요.")
    
    if not 1.96 <= times[-1] <= 2.00:
        raise ValueError("측정 시간이 올바르지 않습니다. 다시 측정하세요.")
    
    # 첫 시점부터 마지막 시점까지 균일한 100개 시점 생성
    target_times = np.linspace(
        times[0], times[-1], SAMPLE_COUNT
    )

    # 시간 열을 제외한 센서 6개 열을 보간
    features = np.column_stack([
        np.interp(target_times, times, raw[:, col])
        for col in range(1, 7)
    ])

    return features.astype(np.float32)
    
class SensorClient:
    def __init__(self, ip, port=5000):
        self.sock = socket.create_connection(
            (ip, port), timeout=10
        )
        self.sock.settimeout(10)
        self.reader = self.sock.makefile(
            'r', encoding='ascii', newline='\n'
        )

    def _readline(self):
        line = self.reader.readline()

        if not line:
            raise ConnectionError("ESP32 disconnected")
        
        return line.strip()
    
    def record(self):
        self.sock.sendall(b"GO\n")

        header = self._readline()
        if header != f"BEGIN,{SAMPLE_COUNT}":
            raise RuntimeError(f"Unexpected header: {header}")

        rows = []

        for _ in range(SAMPLE_COUNT):
            row = [float(v) for v in self._readline().split(",")]

            if len(row) != 7:
                raise RuntimeError("Invalid sensor row.")

            rows.append(row)

        if self._readline() != "END":
            raise RuntimeError("Missing END marker.")

        raw = np.asarray(rows, dtype=np.float64)

        # 저장 전에 측정 품질 확인
        to_features(raw)
        return raw
    
    def close(self):
        self.reader.close()
        self.sock.close()

def countdown():
    for value in (3, 2, 1):
        print(value, flush=True)
        time.sleep(1)

    print("Go! Draw ONE shape now", flush=True)