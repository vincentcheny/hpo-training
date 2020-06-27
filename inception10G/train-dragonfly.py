import os, sys
import numpy as np
import pandas as pd
import skimage.io
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image
import cv2
from sklearn.utils import class_weight, shuffle
import warnings

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import backend as K
import tensorflow.keras
from tensorflow.keras.models import Model
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from dragonfly import load_config, multiobjective_maximise_functions,multiobjective_minimise_functions
from dragonfly import maximise_function,minimise_function
import shutil
import os
import time


class data_generator:
	def create_train(dataset_info, batch_size, shape, augument=True):
		assert shape[2] == 3
		while True:
			dataset_info = shuffle(dataset_info)
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
	bn = BatchNormalization()(input_tensor)
	x = base_model(bn)
	x = Conv2D(32, kernel_size=(1,1), activation='relu')(x)
	x = Flatten()(x)
	x = Dropout(mp[3])(x)
	x = Dense(mp[4], activation='relu')(x)
	x = Dropout(mp[3])(x)
	output = Dense(n_out, activation='sigmoid')(x)
	model = Model(input_tensor, output)
	return model


def runtime_eval(x):
	start = time.time()
	# split data into train, valid
	indexes = np.arange(train_dataset_info.shape[0])
	np.random.shuffle(indexes)
	train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=8)

	# create train and valid datagens
	train_generator = data_generator.create_train(
	    train_dataset_info[train_indexes], x[0], (SIZE,SIZE,3), augument=True)
	validation_generator = data_generator.create_train(
	    train_dataset_info[valid_indexes], 32, (SIZE,SIZE,3), augument=False)


	model = create_model(
	    input_shape=(SIZE,SIZE,3), 
	    n_out=28,
	    mp=x)

	for layer in model.layers:
		layer.trainable = True


	sess =  tf.compat.v1.Session(config=tf.compat.v1.ConfigProto( 
	inter_op_parallelism_threads=x[5],
	intra_op_parallelism_threads=x[6],
	graph_options=tf.compat.v1.GraphOptions(
	    build_cost_model=x[7],
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


	model.compile(loss='binary_crossentropy',
	            optimizer=Adam(lr=x[1]),
	            metrics=['accuracy'])
	his = model.fit(
	    train_generator,
	    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(x[0])),
	    validation_data=validation_generator,
	    validation_steps=np.ceil(float(len(valid_indexes)) / float(x[0])),
	    epochs=x[2], 
	    verbose=2)
	end = time.time()
	
	global final_acc
	final_acc = his.history['val_accuracy'][x[2]- 1]
	
	return float(start - end)


def acc_eval(x):
	global final_acc
	return float(final_acc)



# Load dataset info
path_to_train = '/lustre/project/EricLo/chen.yu/human-protein-data/train/'
data = pd.read_csv('/lustre/project/EricLo/chen.yu/human-protein-data/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)


SIZE = 299
final_acc = 0.0

#model para
batch_list = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
LR_list = [1e-6,2e-6,3e-6,4e-6,5e-6,6e-6,7e-6,8e-6,9e-6,1e-5,2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2]
epoch_list = [1,2,3,4,5,6,7,8]
dropout_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
dense_list = [64,128,256,512,1024]


#hardware para
inter_list = [1,2,3,4]
intra_list = [2,4,6,8,10,12]
build_cost_model_list = [0,2,4,6,8]
infer_shapes_list = [True,False]
place_pruned_graph_list = [True,False]
enable_bfloat16_sendrecv_list = [True,False]
do_common_subexpression_elimination_list = [True,False]
max_folded_constant_list = [2,4,6,8,10]
do_function_inlining_list = [True,False]
global_jit_level_list = [0,1,2]


domain_vars = [{'type': 'discrete_numeric', 'items': batch_list},
                {'type': 'discrete_numeric', 'items': LR_list},
                {'type': 'discrete_numeric', 'items': epoch_list},
                {'type': 'discrete_numeric', 'items': dropout_list},
                {'type': 'discrete_numeric', 'items': dense_list},
                {'type': 'discrete_numeric', 'items': inter_list},
                {'type': 'discrete_numeric', 'items': intra_list},
                {'type': 'discrete_numeric', 'items': build_cost_model_list},
                {'type': 'discrete', 'items': infer_shapes_list},
                {'type': 'discrete', 'items': place_pruned_graph_list},
                {'type': 'discrete', 'items': enable_bfloat16_sendrecv_list},
                {'type': 'discrete', 'items': do_common_subexpression_elimination_list},
                {'type': 'discrete_numeric', 'items': max_folded_constant_list},
                {'type': 'discrete', 'items': do_function_inlining_list},
                {'type': 'discrete_numeric', 'items': global_jit_level_list}
                ]

config_params = {'domain': domain_vars}
config = load_config(config_params)
max_num_evals = 60*60*10
moo_objectives = [runtime_eval, acc_eval]
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type='realtime',config=config)
f = open("./output.log","w+")
print(pareto_opt_pts,file=f)
print("\n",file=f)
print(pareto_opt_vals,file=f)
print("\n",file=f)
print(history,file=f)



