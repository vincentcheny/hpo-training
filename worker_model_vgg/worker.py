from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.training.savercuhk_context import TFTunerContext

# os.environ["SNOOPER_DISABLED"] = "0"
# import pysnooper

tf.compat.v1.disable_eager_execution()
tfds.disable_progress_bar()
BUFFER_SIZE = 10000
BATCH_SIZE = 32

IMG_SIZE = 160  # All images will be resized to 160x160


def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def train_input_fn(batch_size):
    data = tfds.load('cats_vs_dogs', as_supervised=True)
    train_data = data['train']
    train_data = train_data.map(preprocess).shuffle(500).batch(batch_size)
    return train_data


def input_fn(mode, input_context=None):
    datasets, info = tfds.load(name='mnist',
                               with_info=True,
                               as_supervised=True)
    mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                     datasets['test'])

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    if input_context:
        mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                            input_context.input_pipeline_id)
    return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


workers = ["localhost:12345", "localhost:23456"]
task_index = int(sys.argv[1])
TFTunerContext.init_context(len(workers), task_index)
tf.train.TFTunerContext.init_context(len(workers), task_index)

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': workers
    },
    'task': {'type': 'worker', 'index': task_index}
})

LEARNING_RATE = 1e-4


def model_fn(features, labels, mode):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(160, 160, 3), filters=64, kernel_size=(3, 3), padding="same",
                               activation="relu"),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=4096, activation="relu"),
        tf.keras.layers.Dense(units=4096, activation="relu"),
        tf.keras.layers.Dense(units=2, activation="softmax")
    ])
    logits = model(features, training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)(labels, logits)
    loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(
            loss, tf.compat.v1.train.get_or_create_global_step()))


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

config = tf.estimator.RunConfig(train_distribute=strategy, save_checkpoints_steps=5)

classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='./estimator/multiworker', config=config)
# with pysnooper.snoop('./log/file.log', depth=20):
# while True:

try:
    print("start training and evaluating")
    tf.estimator.train_and_evaluate(
        classifier,
        train_spec=tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(32), max_steps=500),
        eval_spec=tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(32), steps=10)
    )
except Exception as e:
    print("[Important] We catch an exception")
    print(e)
    exit(1)
