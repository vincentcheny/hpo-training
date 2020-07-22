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
from dragonfly import load_config, multiobjective_maximise_functions,multiobjective_minimise_functions
from dragonfly import maximise_function,minimise_function
from dragonfly.utils.option_handler import get_option_specs, load_options

import time
tfds.disable_progress_bar()


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    # image = tf.image.grayscale_to_rgb(image)
    image = image / 255.0
    image = tf.image.resize(image, (IMG_SIZE_LENGTH, IMG_SIZE_WIDTH))
    image = tf.cast(image, tf.float32)
    return image, label


def runtime_eval(x):
    print("selected parameter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(x)
    data = tfds.load('plant_leaves', as_supervised=True, data_dir="../../../data")
    train_data = data['train'] 
    train_data = train_data.map(format_example).batch(x[0])
    model = tf.keras.models.load_model('mobilenet.h5',compile=False)

    for layer in model.layers:
        layer.trainable = True


    sess =  tf.compat.v1.Session(config=tf.compat.v1.ConfigProto( 
        inter_op_parallelism_threads=x[4],
        intra_op_parallelism_threads=x[5],
        graph_options=tf.compat.v1.GraphOptions(
            infer_shapes=x[6],
            place_pruned_graph=x[7],
            enable_bfloat16_sendrecv=x[8],
            optimizer_options=tf.compat.v1.OptimizerOptions(
                do_common_subexpression_elimination=x[9],
                max_folded_constant_in_bytes=x[10],
                do_function_inlining=x[11],
                global_jit_level=x[12]))
    ))
    tf.compat.v1.keras.backend.set_session(sess)

    op_type = x[1]
    if op_type == 'adam':
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE),
        			  optimizer=Adam(lr=x[2]),
        			  metrics=['accuracy'])
    elif op_type == 'sgd':
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE),
        	 		  optimizer=SGD(learning_rate=x[2]),
        	 		  metrics=['accuracy'])
    else:
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE),
        			  optimizer=RMSprop(learning_rate=x[2]),
        			  metrics=['accuracy'])

    start = time.time()
    history = model.fit(train_data,epochs=x[3],verbose=2)
    end = time.time()
    spent_time = (start - end) / 3600.0

    global final_acc
    final_acc = history.history['accuracy'][x[3]-1]
    print("Final accuracy: {}".format(final_acc))

    return float(spent_time)


def acc_eval(x):
    global final_acc
    return float(final_acc)


final_acc = 0.0
IMG_SIZE_LENGTH = 600
IMG_SIZE_WIDTH = 400
SHUFFLE_BUFFER_SIZE = 10
IMG_SHAPE = (IMG_SIZE_LENGTH, IMG_SIZE_WIDTH, 3)




batch_list = [2,4,6,8,10,12,14,16]
optimizer_list = ['adam','sgd','rmsp']
LR_list =  [5e-1,2.5e-1,1e-1,5e-2,2.5e-2,1e-2,5e-3,2.5e-3,1e-3,5e-4,2.5e-4,1e-4,5e-5,2.5e-5,1e-5]
epoch_list = [1,2,3,4,5,6,7,8]


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
    get_option_specs('init_capital_frac', False, 0.075, 'Path to the json or pb config file. '),
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




