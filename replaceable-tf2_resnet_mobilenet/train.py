import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import random

seed_value=0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import *
import nni
tfds.disable_progress_bar()


def get_default_params():
    return {
        "BATCH_SIZE": 4,
        "LEARNING_RATE": 8e-6,
        # 'MODEL': ('mnist', 48, 48, 10),
        'DATASET': ('plant_leaves', 600, 400, 22),
        "OPTIMIZER": "adam",
        'NUM_EPOCH': 2 # 20min/epoch
    }


params = get_default_params()
tuned_params = nni.get_next_parameter()
params.update(tuned_params)

IMG_SIZE_LENGTH = params['DATASET'][1]
IMG_SIZE_WIDTH = params['DATASET'][2]
SHUFFLE_BUFFER_SIZE = 10
IMG_SHAPE = (IMG_SIZE_LENGTH, IMG_SIZE_WIDTH, 3)

tf.compat.v1.flags.DEFINE_enum(
    name='model_name', 
    default='mobilenet', 
    enum_values=["mobilenet", "resnet"], 
    help='specify supported model')
MODEL_NAME = tf.compat.v1.flags.FLAGS.model_name
IS_SAVE_MODEL = False
IS_LOAD_MODEL = True

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    # image = tf.image.grayscale_to_rgb(image)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE_LENGTH, IMG_SIZE_WIDTH))
    image = tf.cast(image, tf.float32)
    return image, label


data = tfds.load(params['DATASET'][0], as_supervised=True, data_dir="../../data")
train_data = data['train'] 
# train_data = train_data.shard(num_shards=100, index=0).map(format_example).shuffle(
#     SHUFFLE_BUFFER_SIZE,seed=0).batch(params['BATCH_SIZE'])
train_data = train_data.map(format_example).batch(params['BATCH_SIZE'])

# there is only 'train' in data.keys()
# eval_data = data['test']
# eval_data = eval_data.map(format_example).batch(params['BATCH_SIZE'])

if IS_LOAD_MODEL:
    if MODEL_NAME in ["mobilenet", "resnet"]:
        model = tf.keras.models.load_model(MODEL_NAME+'.h5',compile=False)
    else:
        print(f"You are trying to load {MODEL_NAME} which is not detected.")
        exit(1)
else:
    if MODEL_NAME == "mobilenet":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights=None)
    elif MODEL_NAME == "resnet":
        base_model = tf.keras.applications.ResNet50(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights=None)
    else:
        print(f"You are trying to use {MODEL_NAME} which is not recognized.")
        exit(1)
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(params['DATASET'][3])
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

op_type = params['OPTIMIZER']
if op_type == 'adam':
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE),
    			  optimizer=Adam(lr=params["LEARNING_RATE"]),
    			  metrics=['accuracy'])
elif op_type == 'sgd':
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE),
    	 		  optimizer=SGD(learning_rate=params["LEARNING_RATE"]),
    	 		  metrics=['accuracy'])
else:
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE),
    			  optimizer=RMSprop(learning_rate=params["LEARNING_RATE"]),
    			  metrics=['accuracy'])


epochs = params['NUM_EPOCH'] if 'TRIAL_BUDGET' not in params.keys() else params["TRIAL_BUDGET"]
history = model.fit(train_data,
                    epochs=epochs)

final_acc = history.history['accuracy'][epochs-1]
nni.report_final_result(final_acc)
print("Final accuracy: {}".format(final_acc))

if IS_SAVE_MODEL:
    if MODEL_NAME == "mobilenet":
        model.save("mobilenet.h5")
    elif MODEL_NAME == "resnet":
        model.save("resnet.h5")