[1] 학습파일 폴더
# 학습파일: D:\kiros_backup\DeepLearn\1_PyCode\1_logicgate_1_AND_1.py
# 딥러닝: https://colab.research.google.com/drive/1ZSmJnR4kmVvS9CiFYNoy-Bm3fdC76gAK#scrollTo=85BjRQoepYTC
# 넘파이/판다스: https://colab.research.google.com/drive/1GpgnMptbBZYhiw8pYWCQSryau-pHTQ7j

[2] 환경 설정
1. TensorFlow 2.x  설치 (Keras는 덤): Windows 기준
가. CPU만 사용 시
# 1) 충돌 제거
python -m pip uninstall -y tensorflow tensorflow-intel keras tf-nightly keras-nightly numpy

# 2) 올바른 버전 설치 (CPU용)
python -m pip install "tensorflow==2.16.1" "numpy<2" --upgrade

# 3) 확인 (인터프리터를 3.10.11로 변경 후 실행)
python - <<'PY'
import tensorflow as tf, numpy as np
print("TF :", tf.__version__)
print("NP :", np.__version__)
print("OK :", tf.reduce_sum(tf.constant([[1.,2.],[3.,4.]])).numpy()) 
#1+2+3+4=10: 텐서플로와 넘파이가 버전 충돌없이 잘 연동되어 계산되는지 확인
PY

# 4) Keras 버전 확인
python -m pip show keras

