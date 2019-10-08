from __future__ import absolute_import, division, print_function
from time import sleep
import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
sys.path.append("..")
import grpc_op


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


tfds.disable_progress_bar()
# tf.compat.v1.disable_eager_execution()
BUFFER_SIZE = 10000
BATCH_SIZE = 32

NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = 32 * NUM_WORKERS

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:2001", "localhost:2002"]
    },
    'task': {'type': 'worker', 'index': 1}
})
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
train_datasets_unbatched = datasets['train'].map(scale).shuffle(BUFFER_SIZE)
train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)
run_first_time = True


class callbacktest(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        global run_first_time
        if run_first_time:
            run_first_time = False
            sleep(1)
        grpc_op.clear(multi_worker_model.trainable_variables)

        grpc_op.pull(multi_worker_model.trainable_variables)


callbacks = [callbacktest()]

with strategy.scope():
    # focus on the accuracy change.
    multi_worker_model = build_and_compile_cnn_model()
    multi_worker_model.fit(x=train_datasets,
                           epochs=9,
                           steps_per_epoch=100,
                           callbacks=callbacks)
