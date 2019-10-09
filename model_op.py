import tensorflow as tf
import tensorflow_datasets as tfds
import os
import json


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model


def set_environment(addr1="localhost:2001", port2="localhost:2002", index=0):
    tfds.disable_progress_bar()
    # tf.compat.v1.disable_eager_execution()

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': [addr1, port2]
        },
        'task': {'type': 'worker', 'index': index}
    })
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    return strategy


def prepare_datasets():
    BUFFER_SIZE = 10000
    # BATCH_SIZE = 32

    NUM_WORKERS = 2
    GLOBAL_BATCH_SIZE = 32 * NUM_WORKERS
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    train_datasets_unbatched = datasets['train'].map(scale).shuffle(BUFFER_SIZE)
    train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)
    return train_datasets
