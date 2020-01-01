from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from numpy.random import seed
from tensorflow.keras.initializers import glorot_uniform

seed(0)
tf.random.set_random_seed(0)
tf.compat.v1.disable_eager_execution()
tfds.disable_progress_bar()
BUFFER_SIZE = 10000
BATCH_SIZE = 32

IMG_SIZE = 160  # All images will be resized to IMG_SIZE*IMG_SIZE
IMG_CLASS = 10

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def train_input_fn(batch_size, dataset):
    data = tfds.load(dataset, as_supervised=True)
    train_data = data['train']
    train_data = train_data.map(preprocess).batch(batch_size)
    return train_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('worker', '["localhost:12345", "localhost:23456"]', 'specify workers in the cluster')
tf.app.flags.DEFINE_string('dataset', 'cats_vs_dogs', 'specify dataset')
tf.app.flags.DEFINE_integer('task_index', 0, 'task_index')
tf.app.flags.DEFINE_string('model_dir', './estimator', 'model_dir')
tf.app.flags.DEFINE_integer('save_ckpt_steps', 100, 'save ckpt per n steps')
tf.app.flags.DEFINE_boolean('use_original_ckpt', False, 'use original ckpt')

worker = FLAGS.worker.split(',')
task_index = FLAGS.task_index
model_dir = FLAGS.model_dir
if not FLAGS.use_original_ckpt:
    tf.train.TFTunerContext.init_context(len(worker), task_index)

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': worker
    },
    'task': {'type': 'worker', 'index': task_index}
})

LEARNING_RATE = 0.1 # 1e-2

def model_fn(features, labels, mode):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(160, 160, 3), filters=64, kernel_size=(3, 3), padding="same",
                               activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.Dense(units=256, activation="relu",kernel_initializer='zeros'),
        tf.keras.layers.Dense(units=5, activation="softmax")
    ])
        
    logits = model(features, training=True)

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

config = tf.estimator.RunConfig(save_summary_steps=1, train_distribute=strategy, save_checkpoints_steps=FLAGS.save_ckpt_steps, log_step_count_steps=1)
classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=model_dir, config=config)

tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(32, FLAGS.dataset), max_steps=500),
    eval_spec=tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(32, FLAGS.dataset), steps=10)
)
