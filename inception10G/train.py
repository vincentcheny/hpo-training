seed_value=0
import os, sys
os.environ['PYTHONHASHSEED']=str(seed_value)
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)
import random
random.seed(seed_value)

import pandas as pd
from imgaug import augmenters as iaa
import cv2
from PIL import Image
from sklearn.utils import shuffle

from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam,SGD,RMSprop 
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
import nni


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


def create_model(input_shape, n_out):
	input_tensor = Input(shape=input_shape)
	base_model = InceptionV3(include_top=False,
	               #weights='imagenet',
	               weights=None,
	               input_shape=input_shape)
	#base_model.save('save_model.h5')
	load_model = tf.keras.models.load_model('save_model.h5',compile=False)
	bn = BatchNormalization()(input_tensor)
	#x = base_model(bn)
	x = load_model(bn)
	x = Conv2D(params['FILTERS'], kernel_size=(params['KERNEL_SIZE'],params['KERNEL_SIZE']), activation='relu',kernel_initializer='zeros')(x)
	x = Flatten()(x)
	#x = Dropout(params['DROP_OUT'])(x)
	x = Dense(params['DENSE_UNIT'], activation='relu',kernel_initializer='zeros')(x)
	#x = Dropout(params['DROP_OUT'])(x)
	output = Dense(n_out, activation='sigmoid')(x)
	model = Model(input_tensor, output)
	return model



def main():
	print(params)
	train_dataset_info = []
	for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
		train_dataset_info.append({
			'path':os.path.join(path_to_train, name),
			'labels':np.array([int(label) for label in labels])})
	train_dataset_info = np.array(train_dataset_info)
	#print(len(train_dataset_info))
	
	# split data into train, valid
	indexes = np.arange(train_dataset_info.shape[0])
	#np.random.shuffle(indexes)
	train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=0)

	# create train and valid datagens
	train_generator = data_generator.create_train(
	    train_dataset_info[train_indexes], params['BATCH_SIZE'], (SIZE,SIZE,3), augument=False)
	validation_generator = data_generator.create_train(
	    train_dataset_info[valid_indexes], 10, (SIZE,SIZE,3), augument=False)

	model = create_model(input_shape=(SIZE,SIZE,3), n_out=28)
	#model = tf.keras.models.load_model('model_weight.h5',compile=False)
	for layer in model.layers:
		layer.trainable = True
	
	if params['OPTIMIZER'] == 'adam':
		model.compile(loss='binary_crossentropy',
	            optimizer=Adam(lr=params['LEARNING_RATE']),
	            metrics=['accuracy'])
	elif params['OPTIMIZER'] == 'sgd':
                model.compile(loss='binary_crossentropy',
                    optimizer=SGD(learning_rate=params['LEARNING_RATE']),
                    metrics=['accuracy'])
	else:
                model.compile(loss='binary_crossentropy',
                    optimizer=RMSprop(learning_rate=params['LEARNING_RATE']),
                    metrics=['accuracy'])
	his = model.fit(
	    train_generator,
	    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(params['BATCH_SIZE'])),
	    #steps_per_epoch=10,
	    validation_data=validation_generator,
	    validation_steps=np.ceil(float(len(valid_indexes)) / float(params['BATCH_SIZE'])),
	    #validation_steps=10,
	    epochs=epoch, 
	    verbose=1)
	
	final_acc = his.history['val_accuracy'][epoch-1]
	nni.report_final_result(final_acc)
	#model.save('model_weight.h5')

def get_default_params():
	return {
	'BATCH_SIZE':2,
	'LEARNING_RATE':5e-5,
        'NUM_EPOCH':1,
	"DENSE_UNIT":512,
        'FILTERS':16,
        'KERNEL_SIZE':1,
	'OPTIMIZER':'rmsp'
    }

if __name__ == '__main__':
	params = get_default_params()
	tuned_params = nni.get_next_parameter()
	params.update(tuned_params)
	path_to_train = '../../data/human_protein/train/'
	data = pd.read_csv('../../data/human_protein/train.csv')
	SIZE = 299
	epoch = params['TRIAL_BUDGET'] if 'TRIAL_BUDGET' in params.keys() else params['NUM_EPOCH'] # Assume max TRIAL_BUDGET is 70
	main()

