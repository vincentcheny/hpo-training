from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import datasets, layers, models, losses, optimizers, metrics, preprocessing, utils, callbacks
import numpy as np
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
    x = layers.Dropout(0.5, name='drop9')(x)

    x = layers.Convolution2D(
        classes, (1, 1), padding='valid', name='conv10')(x)
    x = layers.Activation('relu', name='relu_conv10')(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Activation('softmax', name='loss')(x)

    model = models.Model(img_input, out, name='squeezenet')

    return model


def compile_model(model):
    loss = losses.categorical_crossentropy
    optimizer = optimizers.RMSprop(lr=0.0001)
    metric = [metrics.categorical_accuracy, metrics.top_k_categorical_accuracy]
    model.compile(optimizer, loss, metric)
    return model


def main():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    train_datagen = preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
    test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)

    y_train = utils.to_categorical(y_train, num_classes=10)
    y_test = utils.to_categorical(y_test, num_classes=10)

    train_generator = train_datagen.flow(
        x=x_train, y=y_train, batch_size=32, shuffle=True)
    test_generator = test_datagen.flow(
        x=x_test, y=y_test, batch_size=32, shuffle=True)

    sn = SqueezeNet()
    sn = compile_model(sn)
    history = sn.fit_generator(
        train_generator,
        steps_per_epoch=400,
        epochs=params['NUM_EPOCH'],
        validation_data=test_generator,
        validation_steps=200)

    final_acc = history.history['val_categorical_accuracy'][params['NUM_EPOCH'] - 1]
    print("Final accuracy: {}".format(final_acc))
    nni.report_final_result(final_acc)


def get_default_params():
    return {
        "BATCH_SIZE": 16,
        "LEARNING_RATE": 1e-4,
        "DENSE_UNIT": 32,
        "NUM_EPOCH": 1,
        "DROP_OUT": 0.3,
        "inter_op_parallelism_threads": 1,
        "intra_op_parallelism_threads": 2,
        "max_folded_constant": 6,
        "build_cost_model": 4,
        "do_common_subexpression_elimination": 1,
        "do_function_inlining": 1,
        "global_jit_level": 1,
        "infer_shapes": 1,
        "place_pruned_graph": 1,
        "enable_bfloat16_sendrecv": 1
    }


if __name__ == '__main__':
    params = get_default_params()
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)
    main()
