"""Keras의 Sequential / Functional API 모델 생성."""
import tensorflow as tf


def build_sequential():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(16),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(3, activation='softmax'),
    ], name='iris_sequential')


def build_functional():
    inputs = tf.keras.Input(shape=(4,))
    x = tf.keras.layers.Dense(16)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(8)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='iris_functional')
