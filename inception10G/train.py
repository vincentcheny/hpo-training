import os, sys
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
import cv2
from PIL import Image
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
import nni


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


def create_model(input_shape, n_out):
	input_tensor = Input(shape=input_shape)
	base_model = InceptionV3(include_top=False,
	               # weights='imagenet',
	               weights=None,
	               input_shape=input_shape)
	bn = BatchNormalization()(input_tensor)
	x = base_model(bn)
	x = Conv2D(32, kernel_size=(1,1), activation='relu')(x)
	x = Flatten()(x)
	x = Dropout(params['DROP_OUT'])(x)
	x = Dense(params['DENSE_UNIT'], activation='relu')(x)
	x = Dropout(params['DROP_OUT'])(x)
	output = Dense(n_out, activation='sigmoid')(x)
	model = Model(input_tensor, output)
	return model


def get_config():
    return tf.compat.v1.ConfigProto( 
        inter_op_parallelism_threads=int(params['inter_op_parallelism_threads']),
        intra_op_parallelism_threads=int(params['intra_op_parallelism_threads']),
        graph_options=tf.compat.v1.GraphOptions(
            build_cost_model=int(params['build_cost_model']),
            infer_shapes=params['infer_shapes'],
            place_pruned_graph=params['place_pruned_graph'],
            enable_bfloat16_sendrecv=params['enable_bfloat16_sendrecv'],
            optimizer_options=tf.compat.v1.OptimizerOptions(
                do_common_subexpression_elimination=params['do_common_subexpression_elimination'],
                max_folded_constant_in_bytes=int(params['max_folded_constant']),
                do_function_inlining=params['do_function_inlining'],
                global_jit_level=params['global_jit_level'])))


def main(epoch):
	train_dataset_info = []
	for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
		train_dataset_info.append({
			'path':os.path.join(path_to_train, name),
			'labels':np.array([int(label) for label in labels])})
	train_dataset_info = np.array(train_dataset_info)
	
	# split data into train, valid
	indexes = np.arange(train_dataset_info.shape[0])
	np.random.shuffle(indexes)
	train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=8)

	# create train and valid datagens
	train_generator = data_generator.create_train(
	    train_dataset_info[train_indexes], params['BATCH_SIZE'], (SIZE,SIZE,3), augument=True)
	validation_generator = data_generator.create_train(
	    train_dataset_info[valid_indexes], 32, (SIZE,SIZE,3), augument=False)

	model = create_model(
	    input_shape=(SIZE,SIZE,3), 
	    n_out=28)

	for layer in model.layers:
		layer.trainable = True

	sess = tf.compat.v1.Session(config=get_config())
	tf.compat.v1.keras.backend.set_session(sess)

	model.compile(loss='binary_crossentropy',
	            optimizer=Adam(lr=params['LEARNING_RATE']),
	            metrics=['accuracy'])
	his = model.fit(
	    train_generator,
	    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(params['BATCH_SIZE'])),
	    validation_data=validation_generator,
	    validation_steps=np.ceil(float(len(valid_indexes)) / float(params['BATCH_SIZE'])),
	    epochs=epoch, 
	    verbose=2)
	
	final_acc = his.history['val_accuracy'][epoch-1]
	nni.report_final_result(final_acc)

def get_default_params():
	return {
        "BATCH_SIZE": 10,
        "LEARNING_RATE": 1e-4,
        'NUM_EPOCH': 2,
		"DROP_OUT":0.3,
		"DENSE_UNIT":128,
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
	path_to_train = '/lustre/project/EricLo/chen.yu/human-protein-data/train/'
	data = pd.read_csv('/lustre/project/EricLo/chen.yu/human-protein-data/train.csv')
	SIZE = 299
	epoch = (params['TRIAL_BUDGET'] // 10 + 1) if 'TRIAL_BUDGET' in params.keys() else params['NUM_EPOCH'] # Assume max TRIAL_BUDGET is 70
	main(epoch)

