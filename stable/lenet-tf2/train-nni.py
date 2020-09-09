import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_path', type=str, default='./temp.log')
parser.add_argument('--is_soo', type=int, default=1)
args = parser.parse_args()

seed_value= args.seed
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
np.random.seed(seed_value)
import random
random.seed(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)

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
    def __init__(self, filter1,filter2,dense):
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
        self.conv1 = Conv2D(filters=filter1, kernel_size=(9,9), kernel_initializer=my_init,activation='relu')
        self.pool1 = MaxPool2D(pool_size=2)
        self.conv2 = Conv2D(filters=filter2, kernel_size=(5,5), kernel_initializer=my_init,activation='relu')
        self.pool2 = MaxPool2D(pool_size=2)
        self.flatten = Flatten()
        self.fc1 = Dense(units=dense, activation='relu',kernel_initializer=my_init)
        self.fc2 = Dense(units=10, activation='softmax')

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
    """Download and reformat cifar10 dataset"""
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def main():
    print(f'Trial Config:{params}')
    (x_train, y_train), (x_test, y_test) = load_dataset()
    start_time = time.time()
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = create_model(filter1=params['N1'],filter2=params['N2'],dense=params['DENSE_UNIT'])
        op_type = params['OPTIMIZER']
        if op_type == 'adam':
            model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=Adam(lr=params['LEARNING_RATE']),
                    metrics=['accuracy', 'top_k_categorical_accuracy'])
        elif op_type == 'sgd':
            model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=SGD(learning_rate=params['LEARNING_RATE']),
                    metrics=['accuracy', 'top_k_categorical_accuracy'])
        else:
            model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=RMSprop(learning_rate=params['LEARNING_RATE']),
                    metrics=['accuracy', 'top_k_categorical_accuracy'])

    batch_size = params['BATCH_SIZE'] * NUM_GPU
    
    his = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epoch,
        verbose=0,
        validation_data=(x_test, y_test)
    )
    
    end_time = time.time()
    spent_time = (start_time - end_time) / 3600.0
    train_acc = 0. if len(his.history['accuracy']) < 1 else his.history['accuracy'][-1]
    train_top5_acc = 0. if len(his.history['top_k_categorical_accuracy']) < 1 else his.history['top_k_categorical_accuracy'][-1]
    val_acc = 0. if len(his.history['val_accuracy']) < 1 else his.history['val_accuracy'][-1]
    val_top5_acc = 0. if len(his.history['val_top_k_categorical_accuracy']) < 1 else his.history['val_top_k_categorical_accuracy'][-1]
    with open(args.log_path,"a") as f:
        print(train_acc, train_top5_acc, val_acc, val_top5_acc, spent_time*60, start_time, end_time, params,file=f)
    if args.is_soo:
        nni.report_final_result(val_acc)
    else:
        report_dict = {'default':val_acc,'accuracy':val_acc,'runtime':spent_time}
        nni.report_final_result(report_dict)


def get_default_params():
    return {
        'BATCH_SIZE':500,
        'N1':10,
        'N2':50,
        'OPTIMIZER':"adam",
        'LEARNING_RATE':0.0006,
        'NUM_EPOCH':500,
        'DENSE_UNIT':50
        }


if __name__ == '__main__':
    params = get_default_params()
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)
    NUM_GPU = 2
    epoch = params['TRIAL_BUDGET'] if 'TRIAL_BUDGET' in params.keys() else params['NUM_EPOCH']
    main()