from __future__ import absolute_import, division, print_function
from time import sleep
# import grpc
# import transfer_pb2
# import transfer_pb2_grpc
import tensorflow as tf
import os
import json
import tensorflow_datasets as tfds
import pickle
import datetime


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
tf.compat.v1.disable_eager_execution()
BUFFER_SIZE = 10000
BATCH_SIZE = 32

NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = 32 * NUM_WORKERS

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:2001", "localhost:2002"]
    },
    'task': {'type': 'worker', 'index': 0}
})
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
train_datasets_unbatched = datasets['train'].map(scale).shuffle(BUFFER_SIZE)
train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)


class callbacktest(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        tf.summary.scalar('batch loss', data=logs.get('loss'), step=batch)

        multi_worker_model.save_weights(checkpoint_dir)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        multi_worker_model.load_weights(latest)

    # def on_epoch_end(self, epoch, logs=None):
        # print("\nSend GRPC request")

        # grpc_push(multi_worker_model.trainable_variables)
        # grpc_clear(multi_worker_model.trainable_variables)
        # sleep(1)
        # grpc_pull(multi_worker_model.trainable_variables)
        # multi_worker_model.save_weights(checkpoint_dir)
        # latest = tf.train.latest_checkpoint(checkpoint_dir)
        # multi_worker_model.load_weights(latest)


log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
file_writer = tf.summary.create_file_writer(log_dir + "\\metrics")
file_writer.set_as_default()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

callbacks = [callbacktest(), cp_callback, tensorboard_callback]

with strategy.scope():
    # focus on the accuracy change.
    multi_worker_model = build_and_compile_cnn_model()
    # multi_worker_model.save_weights(checkpoint_path.format(epoch=0))
    multi_worker_model.fit(x=train_datasets,
                           epochs=9,
                           steps_per_epoch=100,
                           callbacks=callbacks)
