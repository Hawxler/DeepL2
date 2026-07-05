# 학습파일: D:\kiros_backup\DeepLearn\1_PyCode\1_logicgate_1_AND_1.py
# 딥러닝: https://colab.research.google.com/drive/1ZSmJnR4kmVvS9CiFYNoy-Bm3fdC76gAK#scrollTo=85BjRQoepYTC
# 넘파이/판다스: https://colab.research.google.com/drive/1GpgnMptbBZYhiw8pYWCQSryau-pHTQ7j
import numpy as np
import pandas as pd

s = pd.Series([10, 20, 30, 40])
print(s)

df = pd.DataFrame({
    "name": ["Tom", "Jane", "Mike"],
    "score": [80, 95, 70],
    "age": [20, 21, 19]
})

print(df)