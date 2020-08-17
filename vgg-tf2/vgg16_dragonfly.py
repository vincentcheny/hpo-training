seed_value= 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import numpy as np
np.random.seed(seed_value)
import random
random.seed(seed_value)
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
from dragonfly import load_config, multiobjective_maximise_functions,multiobjective_minimise_functions
from dragonfly import maximise_function,minimise_function
from dragonfly.utils.option_handler import get_option_specs, load_options
import shutil
import time

def my_init(shape, dtype=None):
	return tf.random.normal(shape, dtype=dtype)


##model def
def model_fn(x):
	model = Sequential()
	weight_decay = x[6]
	x_shape = [32,32,3]

	model.add(Conv2D(x[4], (x[5],x[5]), padding='same', input_shape=x_shape, kernel_initializer=my_init, kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Conv2D(x[4], (x[5], x[5]), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(2*x[4], (x[5], x[5]), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Conv2D(2*x[4], (x[5], x[5]), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(4*x[4], (x[5], x[5]), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())


	model.add(Conv2D(4*x[4], (x[5], x[5]), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())


	model.add(Conv2D(4*x[4], (x[5], x[5]), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(8*x[4], (x[5], x[5]), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())


	model.add(Conv2D(8*x[4], (x[5], x[5]), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())


	model.add(Conv2D(8*x[4], (x[5], x[5]), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(8*x[4], (x[5], x[5]), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Conv2D(8*x[4], (x[5], x[5]), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Conv2D(8*x[4], (x[5], x[5]), padding='same',kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(x[7],kernel_initializer=my_init,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Dense(10))
	model.add(Activation('softmax'))
	return model


def runtime_eval(x):
	print("dragonfly selected params!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	print(x)
	return np.random.rand()
	model = model_fn(x)
	op_type = x[1]
	if op_type == 'adam':
		model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=x[0]),
                    metrics=['accuracy'])
	elif op_type == 'sgd':
		model.compile(loss='categorical_crossentropy',
                    optimizer=SGD(learning_rate=x[0]),
                    metrics=['accuracy'])
	else:
		model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(learning_rate=x[0]),
                    metrics=['accuracy'])


	# sgd = optimizers.SGD(lr=x[0], decay=x[1], momentum=0.9, nesterov=True)
	# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	sess =  tf.compat.v1.Session(config=tf.compat.v1.ConfigProto( 
		inter_op_parallelism_threads=x[8],
		intra_op_parallelism_threads=x[9],
		graph_options=tf.compat.v1.GraphOptions(
		    infer_shapes=x[10],
		    place_pruned_graph=x[11],
		    enable_bfloat16_sendrecv=x[12],
		    optimizer_options=tf.compat.v1.OptimizerOptions(
		        do_common_subexpression_elimination=x[13],
		        max_folded_constant_in_bytes=x[14],
		        do_function_inlining=x[15],
		        global_jit_level=x[16]))
	))
	tf.compat.v1.keras.backend.set_session(sess)


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
	history = model.fit_generator(datagen.flow(x_train, y_train,batch_size=x[2]),
                                  steps_per_epoch=x_train.shape[0] // x[2],
                                  #steps_per_epoch=10,
                                  #epochs = 2,
                                  epochs=x[3],
                                  validation_data=(x_test, y_test),
                                  #validation_steps=10, 
                                  verbose=2)
	end = time.time()
	spent_time = (start - end) / 3600.0
	val_acc = history.history['val_accuracy'][x[3]-1]
	#val_acc = history.history['val_accuracy'][2-1]
	global final_acc
	final_acc = val_acc
	return float(spent_time)

def acc_eval(x):
	return np.random.rand()
	global final_acc
	return float(final_acc)



final_acc = 0.0


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

##default setting, 2 epoch 27%, 5 epoch 53.3%
# learning_rate = 0.1

LR_list =  [5e-1,2.5e-1,1e-1,5e-2,2.5e-2,1e-2,5e-3,2.5e-3,1e-3,5e-4,2.5e-4,1e-4,5e-5,2.5e-5,1e-5]
optimizer_list = ['adam','sgd','rmsp']
batch_list = [8,16,32,64,128]
epoch_list = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
filter_list = [8,16,24,32,40,48,56,64]
KS_list = [1,2,3,4,5]
weight_decay_list = [8e-2,4e-2,1e-2,8e-3,4e-3,1e-3,8e-4,4e-4,1e-4,8e-5,4e-5,1e-5]
dense_list = [64,128,256,512,1024]

#test for OOM
#LR_list = [1e-4]
#optimizer_list = ['adam']
#batch_list = [128]
#epoch_list = [10]
#filter_list = [64]
#KS_list = [5]
#weight_decay_list = [1e-5]
#dense_list = [1024]

#hardware para
inter_list = [1,2,3,4]
intra_list = [2,4,6,8,10,12]
infer_shapes_list = [True,False]
place_pruned_graph_list = [True,False]
enable_bfloat16_sendrecv_list = [True,False]
do_common_subexpression_elimination_list = [True,False]
max_folded_constant_list = [2,4,6,8,10]
do_function_inlining_list = [True,False]
global_jit_level_list = [0,1,2]


domain_vars = [{'type': 'discrete_numeric', 'items': LR_list},
                {'type': 'discrete', 'items': optimizer_list},
                {'type': 'discrete_numeric', 'items': batch_list},
                {'type': 'discrete_numeric', 'items': epoch_list},
                {'type': 'discrete_numeric', 'items': filter_list},
                {'type': 'discrete_numeric', 'items': KS_list},
                {'type': 'discrete_numeric', 'items': weight_decay_list},
                {'type': 'discrete_numeric', 'items': dense_list},
                {'type': 'discrete_numeric', 'items': inter_list},
                {'type': 'discrete_numeric', 'items': intra_list},
                {'type': 'discrete', 'items': infer_shapes_list},
                {'type': 'discrete', 'items': place_pruned_graph_list},
                {'type': 'discrete', 'items': enable_bfloat16_sendrecv_list},
                {'type': 'discrete', 'items': do_common_subexpression_elimination_list},
                {'type': 'discrete_numeric', 'items': max_folded_constant_list},
                {'type': 'discrete', 'items': do_function_inlining_list},
                {'type': 'discrete_numeric', 'items': global_jit_level_list},
                ]

is_load_from = ["file", "var"]
if is_load_from == "file":
	points = [
			[['adam', False, False, True, False, False], [0.01, 16, 28, 8, 2, 0.004, 512, 1, 8, 2, 2]], 
			[['sgd', False, False, True, True, False], [0.0005, 64, 22, 56, 4, 0.08, 512, 2, 12, 6, 0]], 
			[['sgd', True, True, True, False, False], [0.01, 32, 2, 16, 4, 0.008, 128, 3, 12, 8, 1]]]
	vals = [
		[-0.004353399806552463, 0.10000000149011612], 
		[-0.05593207067913479, 0.09719999879598618], 
		[-0.00861075931125217, 0.10000000149011612]]
	true_vals= [
		[-0.004353399806552463, 0.10000000149011612], 
		[-0.05593207067913479, 0.09719999879598618], 
		[-0.00861075931125217, 0.10000000149011612]]
	for i in range(3):
		points += points
		vals += vals
		true_vals += true_vals

	data_to_save = {'points': points,
					'vals': vals,
					'true_vals': true_vals}
	temp_save_path = './dragonfly.saved'
	import pickle
	with open(temp_save_path, 'wb') as save_file_handle:
		pickle.dump(data_to_save, save_file_handle)
	load_args = [
		get_option_specs('progress_load_from', False, temp_save_path,
		'Load progress (from possibly a previous run) from this file.')	
	]
	options = load_options(load_args)
elif is_load_from == "var":
	from argparse import Namespace
	points = [
			[['adam', False, False, True, False, False], [0.01, 16, 28, 8, 2, 0.004, 512, 1, 8, 2, 2]], 
			[['sgd', False, False, True, True, False], [0.0005, 64, 22, 56, 4, 0.08, 512, 2, 12, 6, 0]], 
			[['sgd', True, True, True, False, False], [0.01, 32, 2, 16, 4, 0.008, 128, 3, 12, 8, 1]]]
	vals = [
		[-0.004353399806552463, 0.10000000149011612], 
		[-0.05593207067913479, 0.09719999879598618], 
		[-0.00861075931125217, 0.10000000149011612]]
	true_vals= [
		[-0.004353399806552463, 0.10000000149011612], 
		[-0.05593207067913479, 0.09719999879598618], 
		[-0.00861075931125217, 0.10000000149011612]]
	for i in range(1):
		points += [
			[['adam', False, False, True, False, False], [random.random(), 16, 28, 8, 2, random.random(), 512, 1, 8, 2, 2]], 
			[['sgd', False, False, True, True, False], [random.random(), 64, 22, 56, 4, random.random(), 512, 2, 12, 6, 0]], 
			[['sgd', True, True, True, False, False], [random.random(), 32, 2, 16, 4, random.random(), 128, 3, 12, 8, 1]]]
		vals += [-random.random(), random.random()]
		true_vals += [-random.random(), random.random()]
	import pprint
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(points)
	print(len(points),len(vals),len(true_vals))
	qinfos = []
	for i in range(len(points)):
		pt = points[i]
		val = vals[i]
		true_val = true_vals[i]
		qinfo = Namespace(point=pt, val=val, true_val=true_val)
		qinfos.append(qinfo)
	load_args = [
		get_option_specs('prev_evaluations', False, Namespace(qinfos=qinfos),
    		'Data for any previous evaluations.')	
	]
	options = load_options(load_args)
	# qinfo = Namespace(point=pt, val=val, true_val=true_val)
	# self.options.prev_evaluations.qinfos
	# attr:point, val, true_val
else:
	dragonfly_args = [ 
		get_option_specs('report_results_every', False, 2, 'Path to the json or pb config file. '),
		get_option_specs('init_capital', False, None, 'Path to the json or pb config file. '),
		get_option_specs('init_capital_frac', False, 0.017, 'Path to the json or pb config file. '),
		get_option_specs('num_init_evals', False, 2, 'Path to the json or pb config file. ')]
	options = load_options(dragonfly_args)
config_params = {'domain': domain_vars}
config = load_config(config_params)
max_num_evals = 1 #60 * 60 * 10
moo_objectives = [runtime_eval, acc_eval]
st = time.time()
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type='num_evals',config=config,options=options)
et = time.time()
print(f"runtime: {et-st:.4f}s")
# f = open("./output.log","w+")
# print(pareto_opt_pts,file=f)
# print("\n",file=f)
# print(pareto_opt_vals,file=f)
# print("\n",file=f)
# print(history,file=f)









