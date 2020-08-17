seed_value = 0
import os, sys
os.environ['PYTHONHASHSEED']=str(seed_value)
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)
import random
random.seed(seed_value)

import logging
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPool2D)
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
import nni
import time


def my_init(shape, dtype=None):
    return tf.random.normal(shape, dtype=dtype)


class create_model(Model):
    """
    LeNet-5 Model with customizable hyper-parameters
    """
    def __init__(self, filter1, filter2, dense):
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
            kernel_initializer=my_init
        """
        super().__init__()
        self.conv1 = Conv2D(filters=filter1, kernel_size=(9,9), kernel_initializer=my_init,activation='relu')
        self.pool1 = MaxPool2D(pool_size=2)
        self.conv2 = Conv2D(filters=filter2, kernel_size=(5,5), kernel_initializer=my_init,activation='relu')
        self.pool2 = MaxPool2D(pool_size=2)
        self.flatten = Flatten()
        self.fc1 = Dense(units=dense, kernel_initializer=my_init,activation='relu')
        self.fc2 = Dense(units=10,activation='softmax')

    def call(self, x):
        """Override ``Model.call`` to build LeNet-5 model."""
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)


def load_dataset():
    """Download and reformat MNIST dataset"""
    # mnist = tf.keras.datasets.mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # x_train = x_train[..., tf.newaxis]
    # x_test = x_test[..., tf.newaxis]
    return (x_train, y_train), (x_test, y_test)


def get_default_params():
    return {'batch_size':500,
            'n1':10,
            'n2':50,
            'optimizer':"adam",
            'lr':0.0006,
            'epoch':100,
            'dense_size':50
            }

def get_config():
    return tf.compat.v1.ConfigProto( 
        inter_op_parallelism_threads=int(params['inter_op_parallelism_threads']),
        intra_op_parallelism_threads=int(params['intra_op_parallelism_threads']),
        graph_options=tf.compat.v1.GraphOptions(
            infer_shapes=params['infer_shapes'],
            place_pruned_graph=params['place_pruned_graph'],
            enable_bfloat16_sendrecv=params['enable_bfloat16_sendrecv'],
            optimizer_options=tf.compat.v1.OptimizerOptions(
                do_common_subexpression_elimination=params['do_common_subexpression_elimination'],
                max_folded_constant_in_bytes=int(params['max_folded_constant']),
                do_function_inlining=params['do_function_inlining'],
                global_jit_level=params['global_jit_level'])))

def main(params):
    """
    Main program:
      - Build network
      - Prepare dataset
      - Train the model
      - Report accuracy to tuner
    """
    (x_train, y_train), (x_test, y_test) = load_dataset()

    model = create_model(
        filter1=params['n1'],
        filter2=params['n2'],
        dense = params['dense_size']
    )


    op_type = params['optimizer']
    if op_type == 'adam':
        model.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(lr=params['lr']),
                metrics=['accuracy'])
    elif op_type == 'sgd':
        model.compile(loss='sparse_categorical_crossentropy',
                optimizer=SGD(learning_rate=params['lr']),
                metrics=['accuracy'])
    else:
        model.compile(loss='sparse_categorical_crossentropy',
                optimizer=RMSprop(learning_rate=params['lr']),
                metrics=['accuracy'])

    sess = tf.compat.v1.Session(config=get_config())
    tf.compat.v1.keras.backend.set_session(sess)


    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['LEARNING_RATE'])
    # model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # _logger.info('Model built')
    epochs = params['epoch'] if 'TRIAL_BUDGET' not in params.keys() else params["TRIAL_BUDGET"]*5
    print("this trial's buget !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"Number of epochs:{epochs}")
    st = time.time()
    history = model.fit(
                        x_train,
                        y_train,
                        batch_size=params['batch_size'],
                        epochs=1,#epochs,
                        steps_per_epoch=50000//params['batch_size'],
                        verbose=2,
                        validation_data=(x_test, y_test),
                        validation_steps=10000//params['batch_size'],
                    )
    train_loss = history.history['loss'][-1]#[epochs - 1]
    train_acc = history.history['accuracy'][-1]#[epochs - 1]
    print(f"train_loss:{train_loss},train_acc:{train_acc}")
    val_acc = history.history['val_accuracy'][-1]#[epochs - 1]
    et = time.time()
    runtime = (et-st)/60.0
    report_dict = {'default':val_acc,'accuracy':val_acc,'runtime':runtime}
    nni.report_final_result(report_dict)  # send final accuracy to NNI tuner and web UI


if __name__ == '__main__':
    params = get_default_params()
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)
    print(f"params:{params}")
    main(params)