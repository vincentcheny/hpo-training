# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
NNI example trial code.
- Experiment type: Hyper-parameter Optimization
- Trial framework: Tensorflow v2.x (Keras API)
- Model: LeNet-5
- Dataset: MNIST
"""

import logging

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPool2D)
from tensorflow.keras.optimizers import Adam

import nni

_logger = logging.getLogger('mnist_example')
_logger.setLevel(logging.INFO)


class MnistModel(Model):
    """
    LeNet-5 Model with customizable hyper-parameters
    """
    def __init__(self, filter1, filter2):
        """
        Initialize hyper-parameters.
        Parameters
        ----------
        conv_size : int
            Kernel size of convolutional layers.
        hidden_size : int
            Dimensionality of last hidden layer.
        dropout_rate : float
            Dropout rate between two fully connected (dense) layers, to prevent co-adaptation.
        """
        super().__init__()
        self.conv1 = Conv2D(filters=filter1, kernel_size=(9,9), activation='relu')
        self.pool1 = MaxPool2D(pool_size=2)
        self.conv2 = Conv2D(filters=filter2, kernel_size=(5,5), activation='relu')
        self.pool2 = MaxPool2D(pool_size=2)
        self.flatten = Flatten()
        self.fc1 = Dense(units=50, activation='relu')
        # self.dropout = Dropout(rate=dropout_rate)
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, x):
        """Override ``Model.call`` to build LeNet-5 model."""
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)


class ReportIntermediates(Callback):
    """
    Callback class for reporting intermediate accuracy metrics.
    This callback sends accuracy to NNI framework every 100 steps,
    so you can view the learning curve on web UI.
    If an assessor is configured in experiment's YAML file,
    it will use these metrics for early stopping.
    """
    def on_epoch_end(self, epoch, logs=None):
        """Reports intermediate accuracy to NNI framework"""
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])


def load_dataset():
    """Download and reformat MNIST dataset"""
    # mnist = tf.keras.datasets.mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    return (x_train, y_train), (x_test, y_test)


def get_config():
    return tf.compat.v1.ConfigProto( 
        inter_op_parallelism_threads=int(params['inter_op_parallelism_threads']),
        intra_op_parallelism_threads=int(params['intra_op_parallelism_threads']),
        graph_options=tf.compat.v1.GraphOptions(
            build_cost_model=int(params['build_cost_model']),
            infer_shapes=params['infer_shapes'],
            place_pruned_graph=params['place_pruned_graph'],
            enable_bfloat16_sendrecv=params['enable_bfloat16_sendrecv'],
            optimizer_options=tf.compat.v1.OptimizerOptions(
                do_common_subexpression_elimination=params['do_common_subexpression_elimination'],
                max_folded_constant_in_bytes=int(params['max_folded_constant']),
                do_function_inlining=params['do_function_inlining'],
                global_jit_level=params['global_jit_level'])))


def get_default_params():
    return {
        "BATCH_SIZE":32,
        "LEARNING_RATE":1e-4,
        "DROP_OUT":5e-1,
        "DENSE_UNIT":128,
        "OPTIMIZER":"grad",
        "KERNEL_SIZE":3,
        "NUM_EPOCH":81,
        "inter_op_parallelism_threads":1,
        "intra_op_parallelism_threads":2,
        "max_folded_constant":6,
        "build_cost_model":4,
        "do_common_subexpression_elimination":1,
        "do_function_inlining":1,
        "global_jit_level":1,
        "infer_shapes":1,
        "place_pruned_graph":1,
        "enable_bfloat16_sendrecv":1
    }

def main(params):
    """
    Main program:
      - Build network
      - Prepare dataset
      - Train the model
      - Report accuracy to tuner
    """
    model = MnistModel(
        filter1=params['NKERN1'],
        filter2=params['NKERN1']
    )
    # if params['OPTIMIZER'] == 'adam':
    #     optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['LEARNING_RATE'])
    # elif params['OPTIMIZER'] == 'grad':
    #     optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=params['LEARNING_RATE'])
    # elif params['OPTIMIZER'] == 'rmsp':
    #     optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=params['LEARNING_RATE'])
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['LEARNING_RATE'])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _logger.info('Model built')

    (x_train, y_train), (x_test, y_test) = load_dataset()
    _logger.info('Dataset loaded')

    sess = tf.compat.v1.Session(config=get_config())
    tf.compat.v1.keras.backend.set_session(sess)

    model.fit(
        x_train,
        y_train,
        batch_size=params['BATCH_SIZE'],
        epochs=params['NUM_EPOCH'],
        verbose=0,
        callbacks=[ReportIntermediates()],
        validation_data=(x_test, y_test)
    )
    _logger.info('Training completed')

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    nni.report_final_result(accuracy)  # send final accuracy to NNI tuner and web UI
    _logger.info('Final accuracy reported: %s', accuracy)


if __name__ == '__main__':
    params = get_default_params()

    # fetch hyper-parameters from HPO tuner
    # comment out following two lines to run the code without NNI framework
    tuned_params = nni.get_next_parameter()
    while tuned_params['NKERN1'] > tuned_params['NKERN2']:
        tuned_params = nni.get_next_parameter
    params.update(tuned_params)

    _logger.info('Hyper-parameters: %s', params)
    main(params)