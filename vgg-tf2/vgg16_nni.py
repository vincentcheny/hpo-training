seed_value=0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)

import keras
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.datasets import cifar10
import nni
import shutil
import time


def my_init(shape, dtype=None):
    return tf.random.normal(shape, dtype=dtype)

##model def
def create_model():
    model = Sequential()
    weight_decay = params['weight_decay']
    x_shape = [32,32,3]

    model.add(Conv2D(params['filter_num'], (params['kernel_size'],params['kernel_size']), padding='same', input_shape=x_shape, kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(params['filter_num'], (params['kernel_size'], params['kernel_size']), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(2*params['filter_num'], (params['kernel_size'], params['kernel_size']), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(2*params['filter_num'], (params['kernel_size'], params['kernel_size']), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(4*params['filter_num'], (params['kernel_size'], params['kernel_size']), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(4*params['filter_num'], (params['kernel_size'], params['kernel_size']), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(4*params['filter_num'], (params['kernel_size'], params['kernel_size']), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(8*params['filter_num'], (params['kernel_size'], params['kernel_size']), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(8*params['filter_num'], (params['kernel_size'], params['kernel_size']), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(8*params['filter_num'], (params['kernel_size'], params['kernel_size']), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(8*params['filter_num'], (params['kernel_size'], params['kernel_size']), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(8*params['filter_num'], (params['kernel_size'], params['kernel_size']), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(8*params['filter_num'], (params['kernel_size'], params['kernel_size']), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params['dense_size'],kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

##nni init para:
params = {
	'learning_rate': 0.001,
    'optimizer':'adam',
    'batch_size': 32,
    'epoch':10,
    'filter_num':64,
    'kernel_size':3,
    'weight_decay':5e-4,
    'dense_size': 512
}
tuned_params = nni.get_next_parameter()
params.update(tuned_params)


# initializer = tf.keras.initializers.Constant(3.)
model = create_model()
# model.save_weights(save_format="h5",path="model_weights.h5")
# model.load_weights("model_weights.h5")


if params['optimizer'] == 'adam':
    model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=params['learning_rate']),
                    metrics=['accuracy'])
elif params['optimizer'] == 'sgd':
    model.compile(loss='categorical_crossentropy',
                    optimizer=SGD(learning_rate=params['learning_rate']),
                    metrics=['accuracy'])
else:
    model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(learning_rate=params['learning_rate']),
                    metrics=['accuracy'])


# sgd = optimizers.SGD(lr=params['learning_rate'], decay=params['learning_rate_decay'], momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# sess =  tf.compat.v1.Session(config=tf.compat.v1.ConfigProto( 
# 	inter_op_parallelism_threads=int(params['inter_op_parallelism_threads']),
# 	intra_op_parallelism_threads=int(params['intra_op_parallelism_threads']),
# 	graph_options=tf.compat.v1.GraphOptions(
# 	    build_cost_model=int(params['build_cost_model']),
# 	    infer_shapes=params['infer_shapes'],
# 	    place_pruned_graph=params['place_pruned_graph'],
# 	    enable_bfloat16_sendrecv=params['enable_bfloat16_sendrecv'],
# 	    optimizer_options=tf.compat.v1.OptimizerOptions(
# 	        do_common_subexpression_elimination=params['do_common_subexpression_elimination'],
# 	        max_folded_constant_in_bytes=int(params['max_folded_constant']),
# 	        do_function_inlining=params['do_function_inlining'],
# 	        global_jit_level=params['global_jit_level'])),
# 	gpu_options=tf.compat.v1.GPUOptions(
# 	    allow_growth=True,
# 	    allocator_type=params['allocator_type'],
# 	    deferred_deletion_bytes=params['deferred_deletion_bytes'],
# 	    polling_active_delay_usecs=params['polling_active_delay_usecs']))
# )
# tf.compat.v1.keras.backend.set_session(sess)


datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# (std, mean, and principal components if ZCA whitening is applied).
# start = time.time()
epochs = params['epoch'] if 'TRIAL_BUDGET' not in params.keys() else params["TRIAL_BUDGET"]
history = model.fit_generator(datagen.flow(x_train, y_train,batch_size=params['batch_size']),
                              steps_per_epoch=x_train.shape[0] // params['batch_size'],
                              #steps_per_epoch=10,
                              #epochs = 5,
                              epochs=epochs,
                              validation_data=(x_test, y_test),
                              #validation_steps=10,
                              verbose=2)
# end = time.time()
# spent_time = start - end
val_acc = history.history['val_accuracy'][epochs - 1]
#val_acc = history.history['val_accuracy'][5 - 1]
print("final acc: %.4f" % (val_acc))
nni.report_final_result(val_acc)
# model.save_weights(save_format="h5",path="model_weights.h5")


















