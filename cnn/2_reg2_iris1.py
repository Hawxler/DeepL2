# Keras: Functional API 코드
import tensorflow as tf
import pandas as pd

url = "https://raw.githubusercontent.com/hawklux/DeepLearn/refs/heads/main/1_Data/iris.csv"
data = pd.read_csv(url)
print(data.shape)
print(data.head())
print(data.columns)