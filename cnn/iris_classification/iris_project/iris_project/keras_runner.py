"""두 Keras 모델이 공유하는 학습·평가·추론 함수."""
import tensorflow as tf


def train(model, data, epochs=200, batch_size=16, lr=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
    )
    return model.fit(data.X_train, data.y_train, epochs=epochs,
                     batch_size=batch_size, shuffle=True, verbose=2)


def evaluate(model, data):
    return model.evaluate(data.X_test, data.y_test, verbose=0, return_dict=True)


def predict(model, data, rows):
    probabilities = model.predict(data.transform(rows), verbose=0)
    names = data.class_names[probabilities.argmax(axis=1)]
    return names, probabilities
