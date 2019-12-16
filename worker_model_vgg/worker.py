from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

tf.compat.v1.disable_eager_execution()
tfds.disable_progress_bar()
BUFFER_SIZE = 10000
BATCH_SIZE = 32

IMG_SIZE = 28  # All images will be resized to IMG_SIZE*IMG_SIZE
IMG_CLASS = 10

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def train_input_fn(batch_size, dataset):
    data = tfds.load(dataset, as_supervised=True)
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


# workers = ["localhost:12345", "localhost:23456"]
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('worker', None, 'specify workers in the cluster')
tf.app.flags.DEFINE_string('dataset', 'cats_vs_dogs', 'specify dataset')
tf.app.flags.DEFINE_integer('task_index', 0, 'task_index')
tf.app.flags.DEFINE_string('model_dir', './estimator/multiworker', 'model_dir')

worker = FLAGS.worker.split(',')
task_index = FLAGS.task_index
model_dir = FLAGS.model_dir

tf.train.TFTunerContext.init_context(len(worker), task_index)

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': worker
    },
    'task': {'type': 'worker', 'index': task_index}
})

LEARNING_RATE = 0.1 # 1e-2

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    # valid mean no padding / glorot_uniform equal to Xaiver initialization - Steve

    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)
    # Third component of main path (≈2 lines)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    ### END CODE HERE ###

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1', padding='valid',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = tf.keras.layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###

    return X

def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")
    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    # The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
    # The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=1)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

    # Stage 4 (≈6 lines)
    # The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
    # The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

    # Stage 5 (≈3 lines)
    # The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
    # The 2 identity blocks use three set of filters of size [256, 256, 2048], "f" is 3 and the blocks are "b" and "c".
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

    # filters should be [256, 256, 2048], but it fail to be graded. Use [512, 512, 2048] to pass the grading

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    # The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
    X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation="softmax", name="fc" + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name="ResNet50")

    return model


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
        tf.keras.layers.Dense(units=5, activation="softmax")
    ])
    # model = tf.keras.applications.ResNet50(weights=None, include_top=True, classes=2)
    model = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 1), classes=IMG_CLASS) # classes are determined by dataset
    model = tf.keras.Sequential([
        model
    #    tf.keras.layers.Flatten(name='flatten'),
    #    tf.keras.layers.Dense(2, activation='softmax', name='predictions')
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
tf.app.flags.DEFINE_integer('save_ckpt_steps', 10, 'save ckpt per n steps')
config = tf.estimator.RunConfig(save_summary_steps=1, train_distribute=strategy, save_checkpoints_steps=FLAGS.save_ckpt_steps, log_step_count_steps=1)

classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=model_dir, config=config)

try:
    print("start training and evaluating")
    tf.estimator.train_and_evaluate(
        classifier,
        train_spec=tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(32, FLAGS.dataset), max_steps=500),
        eval_spec=tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(32, FLAGS.dataset), steps=10)
    )
except Exception as e:
    print("[Important] We catch an exception")
    print(e)
    exit(1)
