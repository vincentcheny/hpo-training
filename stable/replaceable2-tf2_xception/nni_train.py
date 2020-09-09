seed_value=1234
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
np.random.seed(seed_value)
import random
random.seed(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)


import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.optimizers import *
# from keras.applications import MobileNet
# from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
import nni
import time

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("../../data/humpback-whale-identification/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%2000 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train


def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder


def _set_l2(model, weight_decay):
    """Add L2 regularization into layers with weights

    Reference: https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.add_loss(lambda: tf.keras.regularizers.l2(weight_decay)(layer.kernel))
        elif isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l2(weight_decay)(layer.kernel))


params = {
    'batch_size': 120,
    'weight_decay': 2e-4,
    'optimizer':'adam',
    'lr':2e-4,
    'epoch': 2
}
tuned_params = nni.get_next_parameter()
params.update(tuned_params)


train_df = pd.read_csv("../../data/humpback-whale-identification/train.csv")
# train_df = train_df[:1000]
X = prepareImages(train_df, train_df.shape[0], "train")
X /= 255.0
y, label_encoder = prepare_labels(train_df['Id'])


# model = MobileNet(input_shape=(100, 100, 3), alpha=1., weights=None, classes=5005)
# model = Xception(input_shape=(100, 100, 3), weights=None, classes=5005)
is_load_model = False
if is_load_model:
    model = tf.keras.models.load_model('Xception.h5',compile=False)
else:
    model = Xception(input_shape=(100, 100, 3), weights=None, classes=y.shape[1])
# model.save('Xception_test.h5')
# print("save model weight finish !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")



for layer in model.layers:
        layer.trainable = True

_set_l2(model, params['weight_decay'])


op_type = params['optimizer']
if op_type == 'adam':
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=params['lr']),metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
elif op_type == 'sgd':
    model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=params['lr']),metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
else:
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=params['lr']),metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

    # model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
    #               metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

epochs = params['epoch'] if 'TRIAL_BUDGET' not in params.keys() else params["TRIAL_BUDGET"]
his = model.fit(X, y, 
            epochs=epochs, 
            # steps_per_epoch=10, 
            batch_size=params['batch_size'], 
            verbose=1)
final_acc = his.history['categorical_accuracy'][epochs - 1]
nni.report_final_result(final_acc)