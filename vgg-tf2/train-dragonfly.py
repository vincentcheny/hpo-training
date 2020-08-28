import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--log_path', type=str, default='./temp.log')
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
from dragonfly import load_config, multiobjective_maximise_functions,multiobjective_minimise_functions
from dragonfly import maximise_function,minimise_function
from dragonfly.utils.option_handler import get_option_specs, load_options
import shutil
import time


NUM_GPU = 2

def my_init(shape, dtype=None):
	return tf.random.normal(shape, dtype=dtype)

# model def
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
	print(f'Trial Config:{x}')
	if x[17] in ["global", "gpu_private", "gpu_shared"]:
		os.environ['TF_GPU_THREAD_MODE'] = x[17]
	cross_device_ops = x[18]
	num_packs = x[19]
	if cross_device_ops == "HierarchicalCopyAllReduce":
		mirrored_strategy = tf.distribute.MirroredStrategy(
			cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=num_packs))
	elif cross_device_ops == "NcclAllReduce":
		mirrored_strategy = tf.distribute.MirroredStrategy(
			cross_device_ops=tf.distribute.NcclAllReduce(num_packs=num_packs))
	else:
		mirrored_strategy = tf.distribute.MirroredStrategy()
	with mirrored_strategy.scope():
		model = model_fn(x)
		op_type = x[1]
		if op_type == 'adam':
			model.compile(loss='categorical_crossentropy',
						optimizer=Adam(lr=x[0]),
						metrics=['accuracy', 'top_k_categorical_accuracy'])
		elif op_type == 'sgd':
			model.compile(loss='categorical_crossentropy',
						optimizer=SGD(learning_rate=x[0]),
						metrics=['accuracy', 'top_k_categorical_accuracy'])
		else:
			model.compile(loss='categorical_crossentropy',
						optimizer=RMSprop(learning_rate=x[0]),
						metrics=['accuracy', 'top_k_categorical_accuracy'])

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
	batch_size = x[2] * NUM_GPU
	his = model.fit(datagen.flow(x_train, y_train,batch_size=batch_size),
						steps_per_epoch=x_train.shape[0] // batch_size,
						#steps_per_epoch=10,
						#epochs = 2,
						epochs=x[3],
						validation_data=(x_test, y_test),
						#validation_steps=10, 
						verbose=2)
	end = time.time()
	global final_acc
	spent_time = (start - end) / 3600.0
	train_acc = 0. if len(his.history['accuracy']) < 1 else his.history['accuracy'][-1]
	train_top5_acc = 0. if len(his.history['top_k_categorical_accuracy']) < 1 else his.history['top_k_categorical_accuracy'][-1]
	val_acc = 0. if len(his.history['val_accuracy']) < 1 else his.history['val_accuracy'][-1]
	val_top5_acc = 0. if len(his.history['val_top_k_categorical_accuracy']) < 1 else his.history['val_top_k_categorical_accuracy'][-1]
	with open(args.log_path,"a") as f:
		print(train_acc, train_top5_acc, val_acc, val_top5_acc, spent_time*60, start, end, x,file=f)
	final_acc = val_acc
	return float(spent_time)

def acc_eval(x):
	global final_acc
	return float(final_acc)

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

final_acc = 0.0
data_path = '/research/dept7/ychen/data/cifar10'
(x_train, y_train), (x_test, y_test) = load_data(data_path)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

##default setting, 2 epoch 27%, 5 epoch 53.3%
# learning_rate = 0.1

LR_list =  [5e-1,2.5e-1,1e-1,5e-2,2.5e-2,1e-2,5e-3,2.5e-3,1e-3,5e-4,2.5e-4,1e-4,5e-5,2.5e-5,1e-5]
optimizer_list = ['adam','sgd','rmsp']
batch_list = [8,16,32,64,128]
epoch_list = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
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

tf_gpu_thread_mode_list = ["global", "gpu_private", "gpu_shared"]
cross_device_ops_list = ["NcclAllReduce","HierarchicalCopyAllReduce"]
num_packs_list = [0,1,2,3,4,5]


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
				{'type': 'discrete', 'items': tf_gpu_thread_mode_list},
				{'type': 'discrete', 'items': cross_device_ops_list},
				{'type': 'discrete_numeric', 'items': num_packs_list}
                ]

dragonfly_args = [ 
	get_option_specs('report_results_every', False, 2, 'Path to the json or pb config file. '),
	get_option_specs('init_capital', False, None, 'Path to the json or pb config file. '),
	get_option_specs('init_capital_frac', False, 0.017, 'Path to the json or pb config file. '),
	get_option_specs('num_init_evals', False, 2, 'Path to the json or pb config file. ')]

options = load_options(dragonfly_args)
config_params = {'domain': domain_vars}
config = load_config(config_params)
max_num_evals = 60 * 60 * 60
moo_objectives = [runtime_eval, acc_eval]
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type='realtime',config=config,options=options)
f = open("./output.log","w+")
print(pareto_opt_pts,file=f)
print("\n",file=f)
print(pareto_opt_vals,file=f)
print("\n",file=f)
print(history,file=f)