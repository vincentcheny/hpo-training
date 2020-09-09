from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import os
seed_value=0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)
from tensorflow.keras import datasets, layers, models, losses, optimizers, metrics, preprocessing, utils, callbacks
import nni


def fire_module(x, fire_id, squeeze=16, expand=64):
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"
    s_id = 'fire' + str(fire_id) + '/'

    channel_axis = 3

    x = layers.Convolution2D(
        squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = layers.Activation('relu', name=s_id + relu + sq1x1)(x)

    left = layers.Convolution2D(
        expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = layers.Activation('relu', name=s_id + relu + exp1x1)(left)

    right = layers.Convolution2D(
        expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = layers.Activation('relu', name=s_id + relu + exp3x3)(right)

    x = layers.concatenate(
        [left, right], axis=channel_axis, name=s_id + 'concat')
    return x


def SqueezeNet(input_shape=(32, 32, 3), classes=10):

    img_input = layers.Input(shape=input_shape)
    x = layers.Convolution2D(64, (3, 3), strides=(
        2, 2), padding='valid', name='conv1')(img_input)
    x = layers.Activation('relu', name='relu_conv1')(x)
    # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = layers.Dropout(0.5, name='drop9',seed=123)(x)

    x = layers.Convolution2D(
        classes, (1, 1), padding='valid', name='conv10')(x)
    x = layers.Activation('relu', name='relu_conv10')(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Activation('softmax', name='loss')(x)

    model = models.Model(img_input, out, name='squeezenet')

    return model


def main():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data() # len(x_train):50000 len(x_test):10000
    train_datagen = preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0,#0.1,
        zoom_range=0,#0.1,
        horizontal_flip=False)#True)
    test_datagen = preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0,#0.1,
        zoom_range=0,#0.1,)
    )

    y_train = utils.to_categorical(y_train, num_classes=10)
    y_test = utils.to_categorical(y_test, num_classes=10)

    train_generator = train_datagen.flow(
        x=x_train, y=y_train, batch_size=params['BATCH_SIZE'], shuffle=False)#True)
    test_generator = test_datagen.flow(
        x=x_test, y=y_test, batch_size=params['BATCH_SIZE'], shuffle=False)#True)

    IS_LOAD_MODEL = False
    if IS_LOAD_MODEL:
        model = tf.keras.models.load_model('squeezenet.h5',compile=False)
    else:
        model = SqueezeNet()
    loss = losses.categorical_crossentropy
    metric = [metrics.categorical_accuracy, metrics.top_k_categorical_accuracy]
    # optimizer = optimizers.RMSprop(lr=0.0001)
    if params['OPTIMIZER'] == 'adam':
        optimizer = optimizers.Adam(learning_rate=params["LEARNING_RATE"])
    elif params['OPTIMIZER'] == 'sgd':
        optimizer = optimizers.SGD(learning_rate=params["LEARNING_RATE"])
    else:
        optimizer = optimizers.RMSprop(learning_rate=params["LEARNING_RATE"])
    model.compile(optimizer, loss, metric)
    epochs = params['NUM_EPOCH'] if 'TRIAL_BUDGET' not in params.keys() else params["TRIAL_BUDGET"]
    history = model.fit(
        x=train_generator,
        steps_per_epoch=len(x_train)//params['BATCH_SIZE']*100,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(x_test),
        verbose=1)

    final_acc = history.history['val_categorical_accuracy'][epochs - 1]
    print("Final accuracy: {}".format(final_acc))
    nni.report_final_result(final_acc)
    # sn.save("squeezenet.h5")


def get_default_params():
    return {
        "BATCH_SIZE": 128,
        "LEARNING_RATE": 1e-4,
        "DENSE_UNIT": 32,
        "NUM_EPOCH": 1, # 810s/epoch
        "DROP_OUT": 0.3,
        "OPTIMIZER": "adam"
    }


if __name__ == '__main__':
    params = get_default_params()
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)
    main()
