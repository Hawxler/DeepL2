import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Input
from tensorflow.keras.models import Model
import numpy as np

def create_model():
    inputs = Input(shape=(10,))
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

data = np.random.random((100, 10))
labels = np.random.randint(2, size=(100,1))
print(data[0])
print(labels[0])

model.fit(data, labels, epochs=10, batch_size=10)
