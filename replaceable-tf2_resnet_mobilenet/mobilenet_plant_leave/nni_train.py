seed_value= 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import numpy as np
np.random.seed(seed_value)
import random
random.seed(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)

import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import *
import nni

import time
tfds.disable_progress_bar()


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    # image = tf.image.grayscale_to_rgb(image)
    image = image / 255.0
    image = tf.image.resize(image, (IMG_SIZE_LENGTH, IMG_SIZE_WIDTH))
    image = tf.cast(image, tf.float32)
    return image, label


params = {
    'batch_size': 16,
    'optimizer':'rmsp',
    'lr':2e-4,
    'epoch': 2
}
tuned_params = nni.get_next_parameter()
params.update(tuned_params)


final_acc = 0.0
IMG_SIZE_LENGTH = 600
IMG_SIZE_WIDTH = 400
SHUFFLE_BUFFER_SIZE = 10
IMG_SHAPE = (IMG_SIZE_LENGTH, IMG_SIZE_WIDTH, 3)

data = tfds.load('plant_leaves', as_supervised=True, data_dir="../../../data")
train_data = data['train'] 

train_data = train_data.map(format_example).batch(params['batch_size'])
model = tf.keras.models.load_model('mobilenet.h5',compile=False)
# base_model = tf.keras.applications.MobileNetV2(
#             input_shape=IMG_SHAPE,
#             include_top=False,
#             weights=None)
# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# prediction_layer = tf.keras.layers.Dense(22)
# model = tf.keras.Sequential([
#         base_model,
#         global_average_layer,
#         prediction_layer
#     ])

for layer in model.layers:
    layer.trainable = True


op_type = params['optimizer']
if op_type == 'adam':
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE),
    			  optimizer=Adam(lr=params['lr']),
    			  metrics=['accuracy'])
elif op_type == 'sgd':
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE),
    	 		  optimizer=SGD(learning_rate=params['lr']),
    	 		  metrics=['accuracy'])
else:
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE),
    			  optimizer=RMSprop(learning_rate=params['lr']),
    			  metrics=['accuracy'])


history = model.fit(train_data,epochs=params['epoch'],verbose=2)
# history = model.fit(train_data,steps_per_epoch=10, epochs=2,verbose=1)

final_acc = history.history['accuracy'][params['epoch']-1]
print("Final accuracy: {}".format(final_acc))
nni.report_final_result(final_acc)














