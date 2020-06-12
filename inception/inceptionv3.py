import numpy as np 
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import cv2

from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

from sklearn.model_selection import train_test_split

import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import nni


class ReportIntermediates(Callback):
    """
    Callback class for reporting intermediate accuracy metrics.
    This callback sends accuracy to NNI framework every 100 steps,
    so you can view the learning curve on web UI.
    If an assessor is configured in experiment's YAML file,
    it will use these metrics for early stopping.
    """
    def on_epoch_end(self, epoch, logs=None):
        """Reports intermediate accuracy to NNI framework"""
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])

def main():
	train_folder = './inceptionv3_dog/train/'

	train_dogs = pd.read_csv('./inceptionv3_dog/labels.csv')

	# Get the top 20 breeds which is what we use in this notebook
	top_breeds = sorted(list(train_dogs['breed'].value_counts().head(80).index))
	train_dogs = train_dogs[train_dogs['breed'].isin(top_breeds)]
	# Get the labels of the top 20
	target_labels = train_dogs['breed']

	# One hot code the labels - need this for the model
	one_hot = pd.get_dummies(target_labels, sparse = True)
	one_hot_labels = np.asarray(one_hot)

	# add the actual path name of the pics to the data set
	train_dogs['image_path'] = train_dogs.apply( lambda x: (train_folder + x["id"] + ".jpg" ), axis=1)

	# Convert the images to arrays which is used for the model. Inception uses image sizes of 299 x 299
	train_data = np.array([img_to_array(load_img(img, target_size=(299, 299))) for img in train_dogs['image_path'].values.tolist()]).astype('float32')

	# Split the data into train and validation. The stratify parm will insure  train and validation  
	# will have the same proportions of class labels as the input dataset.
	x_train, x_validation, y_train, y_validation = train_test_split(train_data, target_labels, test_size=0.1, stratify=np.array(target_labels), random_state=100)

	# Need to convert the train and validation labels into one hot encoded format
	# y_train = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
	# y_validation = pd.get_dummies(y_validation.reset_index(drop=True)).as_matrix()
	y_train = pd.get_dummies(y_train.reset_index(drop=True)).to_numpy()
	y_validation = pd.get_dummies(y_validation.reset_index(drop=True)).to_numpy()

	# Create train generator.
	train_datagen = ImageDataGenerator(rescale=1./255, 
	                                   rotation_range=30, 
	                                   # zoom_range = 0.3, 
	                                   width_shift_range=0.2,
	                                   height_shift_range=0.2, 
	                                   horizontal_flip = 'true')
	train_generator = train_datagen.flow(x_train, y_train, shuffle=False, batch_size=params['BATCH_SIZE'], seed=10)

	# Create validation generator
	val_datagen = ImageDataGenerator(rescale = 1./255)
	val_generator = train_datagen.flow(x_validation, y_validation, shuffle=False, batch_size=32, seed=10)

	# Get the InceptionV3 model so we can do transfer learning
	base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(299, 299, 3))
	# Add a global spatial average pooling layer
	bo = base_model.output
	bo = GlobalAveragePooling2D()(bo)
	# Add a fully-connected layer and a logistic layer with 20 classes 
	#(there will be 120 classes for the final submission)
	bo = Dense(params['DENSE_UNIT'], activation='relu')(bo)
	predictions = Dense(80, activation='softmax')(bo)

	model = Model(inputs = base_model.input, outputs = predictions)

	# first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
	for layer in base_model.layers:
	    # layer.trainable = False
	    layer.trainable = True

	sess = tf.compat.v1.Session(config=get_config())
	tf.compat.v1.keras.backend.set_session(sess)
	
	# Compile with Adam
	model.compile(Adam(lr=params['LEARNING_RATE']), loss='categorical_crossentropy', metrics=['accuracy'])

	# Train the model
	his = model.fit(train_generator,
                      validation_data = val_generator,
                      epochs = params['NUM_EPOCH'],
                      verbose = 2,
					  callbacks=[ReportIntermediates()])
	print("***********************************")
	final_acc = his.history['val_accuracy'][params['NUM_EPOCH']-1]
	print(final_acc)
	nni.report_final_result(final_acc)

def get_default_params():
    return {
        "BATCH_SIZE":10,
        "LEARNING_RATE":1e-4,
        'DENSE_UNIT':32,
        'NUM_EPOCH':1,
		"inter_op_parallelism_threads":1,
        "intra_op_parallelism_threads":2,
        "max_folded_constant":6,
        "build_cost_model":4,
        "do_common_subexpression_elimination":1,
        "do_function_inlining":1,
        "global_jit_level":1,
        "infer_shapes":1,
        "place_pruned_graph":1,
        "enable_bfloat16_sendrecv":1
    }

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

if __name__ == '__main__':
	params = get_default_params()
	tuned_params = nni.get_next_parameter()
	params.update(tuned_params)
	main()
