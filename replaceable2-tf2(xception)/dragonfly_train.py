seed_value= 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import numpy as np
np.random.seed(seed_value)
import random
random.seed(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)


import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import layers
from keras.preprocessing import image
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import *
# from keras.applications import MobileNet
# from keras.applications.mobilenet import preprocess_input
from keras.applications import Xception
from keras.applications.xception import preprocess_input

from dragonfly import load_config, multiobjective_maximise_functions,multiobjective_minimise_functions
from dragonfly import maximise_function,minimise_function
from dragonfly.utils.option_handler import get_option_specs, load_options

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
        if (count%100 == 0):
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



def runtime_eval(x):
    print("selected parameter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(x)
    # model = MobileNet(input_shape=(100, 100, 3), alpha=1., weights=None, classes=5005)
    # model = Xception(input_shape=(100, 100, 3), weights=None, classes=5005)
    # model.save('Xception.h5')
    # print("save model weight finish !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    model = tf.keras.models.load_model('Xception.h5',compile=False)

    for layer in model.layers:
        layer.trainable = True

    _set_l2(model, x[1])

    sess =  tf.compat.v1.Session(config=tf.compat.v1.ConfigProto( 
        inter_op_parallelism_threads=x[5],
        intra_op_parallelism_threads=x[6],
        graph_options=tf.compat.v1.GraphOptions(
            infer_shapes=x[7],
            place_pruned_graph=x[8],
            enable_bfloat16_sendrecv=x[9],
            optimizer_options=tf.compat.v1.OptimizerOptions(
                do_common_subexpression_elimination=x[10],
                max_folded_constant_in_bytes=x[11],
                do_function_inlining=x[12],
                global_jit_level=x[13]))
    ))
    tf.compat.v1.keras.backend.set_session(sess)

    op_type = x[2]
    if op_type == 'adam':
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=x[3]),metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
    elif op_type == 'sgd':
        model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=x[3]),metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
    else:
        model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=x[3]),metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

    # model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
    #               metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
    start = time.time()
    his = model.fit(X, y, 
        epochs=x[4], 
        # steps_per_epoch=10, 
        batch_size=x[0], 
        verbose=2)
    end = time.time()
    spent_time = (end - start + 60.0 * 6.0) / 3600.0
    global final_acc
    final_acc = his.history['categorical_accuracy'][x[4] - 1]
    return -float(spent_time)


def acc_eval(x):
    global final_acc
    return float(final_acc)


final_acc = 0.0

train_df = pd.read_csv("../../data/humpback-whale-identification/train.csv")
# train_df = train_df[:1000]
X = prepareImages(train_df, train_df.shape[0], "train")
X /= 255.0
y, label_encoder = prepare_labels(train_df['Id'])


batch_list = [8,16,24,32,40,48,56,64,72,80,88,96,104,112,120]
weight_decay_list = [5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5]
optimizer_list = ['adam','sgd','rmsp']
LR_list =  [5e-1,2.5e-1,1e-1,5e-2,2.5e-2,1e-2,5e-3,2.5e-3,1e-3,5e-4,2.5e-4,1e-4,5e-5,2.5e-5,1e-5]
epoch_list = [10,11,12,13,14,15,16,17,18,19,20]


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


domain_vars = [{'type': 'discrete_numeric', 'items': batch_list},
                {'type': 'discrete_numeric', 'items': weight_decay_list},
                {'type': 'discrete', 'items': optimizer_list},
                {'type': 'discrete_numeric', 'items': LR_list},
                {'type': 'discrete_numeric', 'items': epoch_list},
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

dragonfly_args = [ 
    get_option_specs('report_results_every', False, 2, 'Path to the json or pb config file. '),
    get_option_specs('init_capital', False, None, 'Path to the json or pb config file. '),
    get_option_specs('init_capital_frac', False, 0.04, 'Path to the json or pb config file. '),
    get_option_specs('num_init_evals', False, 2, 'Path to the json or pb config file. ')]

options = load_options(dragonfly_args)
config_params = {'domain': domain_vars}
config = load_config(config_params)
max_num_evals = 60 * 60 * 10
moo_objectives = [runtime_eval, acc_eval]
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type="realtime",config=config,options=options)
f = open("./output.log","w+")
print(pareto_opt_pts,file=f)
print("\n",file=f)
print(pareto_opt_vals,file=f)
print("\n",file=f)
print(history,file=f)