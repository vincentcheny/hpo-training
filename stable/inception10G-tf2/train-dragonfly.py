import sys
import os
import time
import argparse
import numpy as np
import random
import pandas as pd
import skimage.io
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image
import cv2
import warnings
from sklearn.utils import class_weight, shuffle

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--kernel_initializer', type=str, default='glorot_uniform')
parser.add_argument('--is_load_model', type=int, default=0)
parser.add_argument('--is_augument', type=int, default=1)
parser.add_argument('--log_path', type=str, default='./temp.log')
parser.add_argument('--is_soo', type=int, default=0)

args = parser.parse_args()
print(f"seed:{args.seed} kernel_initializer:{args.kernel_initializer} is_load_model:{args.is_load_model} is_augument:{args.is_augument}")

seed_value = args.seed
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
np.random.seed(seed_value)
random.seed(seed_value)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split

tf.compat.v1.set_random_seed(seed_value)

from dragonfly import load_config, multiobjective_maximise_functions,multiobjective_minimise_functions
from dragonfly import maximise_function,minimise_function
from dragonfly.utils.option_handler import get_option_specs, load_options

SIZE = 299
NUM_GPU = 2

class data_generator:
	def create_train(dataset_info, batch_size, shape, augument=False):
		assert shape[2] == 3
		while True:
			dataset_info = shuffle(dataset_info,random_state=0)
			for start in range(0, len(dataset_info), batch_size):
				end = min(start + batch_size, len(dataset_info))
				batch_images = []
				X_train_batch = dataset_info[start:end]
				batch_labels = np.zeros((len(X_train_batch), 28))
				for i in range(len(X_train_batch)):
				    image = data_generator.load_image(
				        X_train_batch[i]['path'], shape)   
				    if augument:
				        image = data_generator.augment(image)
				    batch_images.append(image/255.)
				    batch_labels[i][X_train_batch[i]['labels']] = 1
				yield np.array(batch_images, np.float32), batch_labels
	def load_image(path, shape):
		image_red_ch = Image.open(path+'_red.png')
		image_yellow_ch = Image.open(path+'_yellow.png')
		image_green_ch = Image.open(path+'_green.png')
		image_blue_ch = Image.open(path+'_blue.png')
		image = np.stack((
		np.array(image_red_ch), 
		np.array(image_green_ch), 
		np.array(image_blue_ch)), -1)
		image = cv2.resize(image, (shape[0], shape[1]))
		return image

	def augment(image):
		augment_img = iaa.Sequential([
	        iaa.OneOf([
	            iaa.Affine(rotate=0),
	            iaa.Affine(rotate=90),
	            iaa.Affine(rotate=180),
	            iaa.Affine(rotate=270),
	            iaa.Fliplr(0.5),
	            iaa.Flipud(0.5),
	        ])], random_order=True)

		image_aug = augment_img.augment_image(image)
		return image_aug


def create_model(input_shape, n_out,mp):
	input_tensor = Input(shape=input_shape)
	base_model = InceptionV3(include_top=False,
	              # weights='imagenet',
	              weights=None,
	              input_shape=input_shape)
	# load_model = tf.keras.models.load_model('save_model.h5',compile=False)
	bn = BatchNormalization()(input_tensor)
	x = base_model(bn)
	# x = load_model(bn)
	x = Conv2D(32, kernel_size=(mp[5],mp[5]), activation='relu',kernel_initializer=args.kernel_initializer)(x)
	x = Flatten()(x)
	
	x = Dropout(mp[3])(x)
	x = Dense(mp[4], activation='relu',kernel_initializer=args.kernel_initializer)(x)
	x = Dropout(mp[3])(x)
	output = Dense(n_out, activation='sigmoid')(x)
	model = Model(input_tensor, output)
	return model


def runtime_eval(x):
	for i in range(len(x)):
		if type(x[i]) is np.ndarray:
			x[i] = x[i][0]
	print(f"[INFO] Trial Config: {x}")
	start = time.time()
	# split data into train, valid
	indexes = np.arange(train_dataset_info.shape[0])
	#np.random.shuffle(indexes)
	train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=0)
	# create train and valid datagens
	is_augument = args.is_augument == 1
	batch_size = x[0] * NUM_GPU
	train_generator = data_generator.create_train(
	    train_dataset_info[train_indexes], batch_size, (SIZE,SIZE,3), augument=is_augument)
	validation_generator = data_generator.create_train(
	    train_dataset_info[valid_indexes], 32, (SIZE,SIZE,3), augument=False)
	if x[16] in ["global", "gpu_private", "gpu_shared"]:
		os.environ['TF_GPU_THREAD_MODE'] = x[16]
	cross_device_ops = x[17]
	num_packs = x[18]
	if cross_device_ops == "HierarchicalCopyAllReduce":
		mirrored_strategy = tf.distribute.MirroredStrategy(
			cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=num_packs))
	elif cross_device_ops == "NcclAllReduce":
		mirrored_strategy = tf.distribute.MirroredStrategy(
			cross_device_ops=tf.distribute.NcclAllReduce(num_packs=num_packs))
	else:
		mirrored_strategy = tf.distribute.MirroredStrategy()
	with mirrored_strategy.scope():
		is_load_model = args.is_load_model
		if is_load_model:
			model = tf.keras.models.load_model("../../save/inception_human_save_tpe")
		else:
			model = create_model(input_shape=(SIZE,SIZE,3), n_out=28,mp=x)
	for layer in model.layers:
		layer.trainable = True
	sess =  tf.compat.v1.Session(config=tf.compat.v1.ConfigProto( 
	inter_op_parallelism_threads=x[6],
	intra_op_parallelism_threads=x[7],
	graph_options=tf.compat.v1.GraphOptions(
	    #build_cost_model=x[7],
	    infer_shapes=x[8],
	    place_pruned_graph=x[9],
	    enable_bfloat16_sendrecv=x[10],
	    optimizer_options=tf.compat.v1.OptimizerOptions(
	        do_common_subexpression_elimination=x[11],
	        max_folded_constant_in_bytes=x[12],
	        do_function_inlining=x[13],
	        global_jit_level=x[14])),
		)
	)
	tf.compat.v1.keras.backend.set_session(sess)
	if x[15] == 'adam':
		model.compile(loss='binary_crossentropy',
                    optimizer=Adam(lr=x[1]),
                    metrics=['accuracy', 'top_k_categorical_accuracy'])
	elif x[15] == 'sgd':
		model.compile(loss='binary_crossentropy',
                    optimizer=SGD(learning_rate=x[1]),
                    metrics=['accuracy', 'top_k_categorical_accuracy'])
	else:
		model.compile(loss='binary_crossentropy',
                    optimizer=RMSprop(learning_rate=x[1]),
                    metrics=['accuracy', 'top_k_categorical_accuracy'])
	his = model.fit(
	    train_generator,
	    steps_per_epoch=np.ceil(float(len(train_indexes)) / batch_size),
	    # steps_per_epoch=10,
	    validation_data=validation_generator,
	    validation_steps=np.ceil(float(len(valid_indexes)) / batch_size),
	    # validation_steps=10,
	    epochs=x[2], 
	    #epochs=5,
	    verbose=2)
	# keras_model_path = "../../save/inception_human_save_glo"
	# model.save(keras_model_path)
	end = time.time()
	global final_acc
	spent_time = (end - start) / 3600.0
	train_acc = 0. if len(his.history['accuracy']) < 1 else his.history['accuracy'][-1]
	train_top5_acc = 0. if len(his.history['top_k_categorical_accuracy']) < 1 else his.history['top_k_categorical_accuracy'][-1]
	val_acc = 0. if len(his.history['val_accuracy']) < 1 else his.history['val_accuracy'][-1]
	val_top5_acc = 0. if len(his.history['val_top_k_categorical_accuracy']) < 1 else his.history['val_top_k_categorical_accuracy'][-1]
	with open(args.log_path,"a") as f:
		print(train_acc, train_top5_acc, val_acc, val_top5_acc, spent_time*(-60), start, end, x, file=f)
	final_acc = val_acc
	return -spent_time

def acc_eval(x):
	global final_acc
	#return float(0.1238)
	return float(final_acc)

# Load dataset info
path_to_train = '/research/dept7/ychen/data/human/train/'
data = pd.read_csv('/research/dept7/ychen/data/human/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)


final_acc = 0.0

#model para
batch_list = [16,32,48,64,80]
LR_list = [1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2]
epoch_list = [1,2,3,4,5]
dense_list = [64,128,256,512]
dropout_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
filter_list = [16,32,48,64,128]
# kernel_list = [1,2,3,4,5]
kernel_list = [1]

#hardware para
inter_list = [1,2,3,4]
intra_list = [2,4,6,8,10,12]
#build_cost_model_list = [0,2,4,6,8]
infer_shapes_list = [True,False]
place_pruned_graph_list = [True,False]
enable_bfloat16_sendrecv_list = [True,False]
do_common_subexpression_elimination_list = [True,False]
max_folded_constant_list = [2,4,6,8,10]
do_function_inlining_list = [True,False]
global_jit_level_list = [0,1,2]
optimizer_list = ['sgd','rmsp','adam']

tf_gpu_thread_mode_list = ["global", "gpu_private", "gpu_shared"]
cross_device_ops_list = ["NcclAllReduce","HierarchicalCopyAllReduce"]
num_packs_list = [0,1,2,3,4,5]

domain_vars = [{'type': 'discrete_numeric', 'items': batch_list},
                {'type': 'discrete_numeric', 'items': LR_list},
                {'type': 'discrete_numeric', 'items': epoch_list},
				{'type': 'discrete_numeric', 'items': dropout_list},
                {'type': 'discrete_numeric', 'items': dense_list},
                {'type': 'discrete_numeric', 'items': kernel_list},
                {'type': 'discrete_numeric', 'items': inter_list},
                {'type': 'discrete_numeric', 'items': intra_list},
                #{'type': 'discrete_numeric', 'items': build_cost_model_list},
                {'type': 'discrete', 'items': infer_shapes_list},
                {'type': 'discrete', 'items': place_pruned_graph_list},
                {'type': 'discrete', 'items': enable_bfloat16_sendrecv_list},
                {'type': 'discrete', 'items': do_common_subexpression_elimination_list},
                {'type': 'discrete_numeric', 'items': max_folded_constant_list},
                {'type': 'discrete', 'items': do_function_inlining_list},
                {'type': 'discrete_numeric', 'items': global_jit_level_list},
				{'type': 'discrete', 'items': optimizer_list},
				{'type': 'discrete', 'items': tf_gpu_thread_mode_list},
				{'type': 'discrete', 'items': cross_device_ops_list},
				{'type': 'discrete_numeric', 'items': num_packs_list}
                ]


dragonfly_args = [ 
	get_option_specs('report_results_every', False, 2, 'Path to the json or pb config file. '),
	get_option_specs('init_capital', False, None, 'Path to the json or pb config file. '),
	# get_option_specs('init_capital_frac', False, 0.07, 'Path to the json or pb config file. '),
	get_option_specs('num_init_evals', False, 2, 'Path to the json or pb config file. ')
]

options = load_options(dragonfly_args)
config_params = {'domain': domain_vars}
config = load_config(config_params)
max_num_evals = 60*60*60
moo_objectives = [runtime_eval, acc_eval]
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type='realtime',config=config,options=options)
f = open("./pareto-output-with-hw.log","w+")
print(pareto_opt_pts,file=f)
print("\n",file=f)
print(pareto_opt_vals,file=f)
print("\n",file=f)
print(history,file=f)



