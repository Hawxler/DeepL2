import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.utils import plot_model # pip install pydot graphviz

url = "https://raw.githubusercontent.com/hawklux/DeepLearn/main/1_Data/lemonade.csv"

data = pd.read_csv(url)
print(data.shape)
print(data.head())

X_features = data[["온도"]]
y_target = data[["판매량"]]
print(X_features.shape, y_target.shape)
print("-------------------")
print("X_features:", X_features)
print("-------------------")


X = tf.keras.layers.Input(shape=[1])
y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, y)
model.compile(optimizer='adam', loss='mse')

model.fit(X_features, y_target, epochs=10, verbose=1)

X_new = np.array([[15.0]], dtype=np.float32)
pred = model.predict(X_new, verbose=1).ravel()[0]
print("예측 판매량 np:", pred)

sales = model.predict(pd.DataFrame({"온도":[15.0]}))
print("예측 판매량 pd:", sales[0][0])
print(sales[0])
print(sales)

plot_model(model)