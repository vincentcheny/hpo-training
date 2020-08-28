import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--seed', type=int, default=123)
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

from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.python.keras.datasets.cifar import load_batch
import nni
import shutil
import time


def my_init(shape, dtype=None):
	return tf.random.normal(shape, dtype=dtype)

# model def
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


def main():
	print(f'Trial Config:{params}')
	mirrored_strategy = tf.distribute.MirroredStrategy()
	with mirrored_strategy.scope():
		model = create_model()
		if params['optimizer'] == 'adam':
			model.compile(loss='categorical_crossentropy',
						optimizer=Adam(lr=params['learning_rate']),
						metrics=['accuracy', 'top_k_categorical_accuracy'])
		elif params['optimizer'] == 'sgd':
			model.compile(loss='categorical_crossentropy',
						optimizer=SGD(learning_rate=params['learning_rate']),
						metrics=['accuracy', 'top_k_categorical_accuracy'])
		else:
			model.compile(loss='categorical_crossentropy',
						optimizer=RMSprop(learning_rate=params['learning_rate']),
						metrics=['accuracy', 'top_k_categorical_accuracy'])

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

	start = time.time()
	batch_size = params['batch_size'] * NUM_GPU
	his = model.fit(datagen.flow(x_train, y_train,batch_size=batch_size),
						steps_per_epoch=x_train.shape[0] // batch_size,
						#steps_per_epoch=10,
						#epochs = 2,
						epochs=epoch,
						validation_data=(x_test, y_test),
						#validation_steps=10, 
						verbose=2)
	end = time.time()
	spent_time = (start - end) / 3600.0
	train_acc = 0. if len(his.history['accuracy']) < 1 else his.history['accuracy'][-1]
	train_top5_acc = 0. if len(his.history['top_k_categorical_accuracy']) < 1 else his.history['top_k_categorical_accuracy'][-1]
	val_acc = 0. if len(his.history['val_accuracy']) < 1 else his.history['val_accuracy'][-1]
	val_top5_acc = 0. if len(his.history['val_top_k_categorical_accuracy']) < 1 else his.history['val_top_k_categorical_accuracy'][-1]
	with open(args.log_path,"a") as f:
		print(train_acc, train_top5_acc, val_acc, val_top5_acc, spent_time*60, start, end, params, file=f)
	if args.is_soo == 1:
		nni.report_final_result(val_acc)
	else:
		report_dict = {'accuracy':val_acc,'runtime':spent_time,'default':val_acc}
		nni.report_final_result(report_dict)


def load_data(path):
	num_train_samples = 50000

	x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
	y_train = np.empty((num_train_samples,), dtype='uint8')

	for i in range(1, 6):
		fpath = os.path.join(path, 'data_batch_' + str(i))
		(x_train[(i - 1) * 10000:i * 10000, :, :, :],
			y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

	fpath = os.path.join(path, 'test_batch')
	x_test, y_test = load_batch(fpath)

	y_train = np.reshape(y_train, (len(y_train), 1))
	y_test = np.reshape(y_test, (len(y_test), 1))

	if K.image_data_format() == 'channels_last':
		x_train = x_train.transpose(0, 2, 3, 1)
		x_test = x_test.transpose(0, 2, 3, 1)

	x_test = x_test.astype(x_train.dtype)
	y_test = y_test.astype(y_train.dtype)

	return (x_train, y_train), (x_test, y_test)

def get_default_params():
    return {
	'learning_rate': 0.001,
    'optimizer':'adam',
    'batch_size': 32,
    'epoch':10,
    'filter_num':64,
    'kernel_size':3,
    'weight_decay':5e-4,
    'dense_size': 512
}


if __name__ == '__main__':
	params = get_default_params()
	tuned_params = nni.get_next_parameter()
	params.update(tuned_params)
	data_path = '/research/dept7/ychen/data/cifar10'
	(x_train, y_train), (x_test, y_test) = load_data(data_path)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	y_train = tf.keras.utils.to_categorical(y_train, 10)
	y_test = tf.keras.utils.to_categorical(y_test, 10)
	NUM_GPU = 2
	epoch = params['TRIAL_BUDGET'] if 'TRIAL_BUDGET' in params.keys() else params['epoch']
	main()