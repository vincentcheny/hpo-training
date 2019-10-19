from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import json
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.client import device_lib
# os.environ["SNOOPER_DISABLED"] = "0"
import pysnooper
import shutil
# if os.path.isdir('./estimator'):
#     shutil.rmtree('./estimator')

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement = True
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)
tfds.disable_progress_bar()
BUFFER_SIZE = 10000
BATCH_SIZE = 64


def input_fn(mode, input_context=None):
    datasets, info = tfds.load(name='mnist',
                               with_info=True,
                               as_supervised=True)
    mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                     datasets['test'])

    # if mode == tf.estimator.ModeKeys.TRAIN:
    #     print("Device info before training:")
    #     print(device_lib.list_local_devices())
    # else:
    #     print("Device info after training:")
    #     print(device_lib.list_local_devices())

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    if input_context:
        mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                            input_context.input_pipeline_id)
    return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 1}
})

LEARNING_RATE = 1e-4


def model_fn(features, labels, mode):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    logits = model(features, training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits)
    loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(
            loss, tf.compat.v1.train.get_or_create_global_step()))


restart = True
# with pysnooper.snoop('./log/file.log', depth=20):
while restart:
    restart = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    config = tf.estimator.RunConfig(train_distribute=strategy)

    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir='./estimator/multiworker', config=config)
    # print("Group Key:", cross_device_utils.CollectiveKeys().get_group_key(["/job:worker/replica:0/task:1/device:CPU:0"]))
    # print(device_lib.list_local_devices())
    try:
        print("start training and evaluating")
        tf.estimator.train_and_evaluate(
            classifier,
            train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
            eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
        )
    except Exception as e:
        print("[Important] We catch an exception")
        print(e)
        # tf.compat.v1.reset_default_graph()
        # tf.compat.v1.Session.reset("grpc://localhost:23456")
        restart = True

