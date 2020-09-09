import tensorflow_datasets as tfds
import tensorflow as tf
from dragonfly import load_config, multiobjective_maximise_functions,multiobjective_minimise_functions
from dragonfly import maximise_function,minimise_function
import shutil
import os
import time


def acc_eval(x):
    global final_acc
    return float(final_acc)

def runtime_eval(x):
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    encoder = info.features['text'].encoder
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(x[0])
    test_dataset = test_dataset.padded_batch(x[0])

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(x[1], activation='relu'),
        tf.keras.layers.Dropout(x[2]),
        tf.keras.layers.Dense(1)
    ])

    print("build model")

    global OPTIMIZER
    OPTIMIZER = x[3]
    if OPTIMIZER == 'adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=x[4])
    elif OPTIMIZER == 'grad':
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=x[4])
    elif OPTIMIZER == 'rmsp':
        optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=x[4])


    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    sess =  tf.compat.v1.Session(config=tf.compat.v1.ConfigProto( 
        inter_op_parallelism_threads=x[6],
        intra_op_parallelism_threads=x[7],
        graph_options=tf.compat.v1.GraphOptions(
            build_cost_model=x[8],
            infer_shapes=x[9],
            place_pruned_graph=x[10],
            enable_bfloat16_sendrecv=x[11],
            optimizer_options=tf.compat.v1.OptimizerOptions(
                do_common_subexpression_elimination=x[12],
                max_folded_constant_in_bytes=x[13],
                do_function_inlining=x[14],
                global_jit_level=x[15]))
        )
    )
    tf.compat.v1.keras.backend.set_session(sess)

    print("start fit")
    start_time = time.time()
    history = model.fit(train_dataset, epochs=x[5],
                        validation_data=test_dataset,
                        verbose=2)


    test_acc = history.history['val_accuracy'][x[5]- 1]

    # test_loss, test_acc = model.evaluate(test_dataset)
    end_time = time.time()

    # print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))
    global final_acc
    final_acc = test_acc
    return float(start_time - end_time)




tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)



BUFFER_SIZE = 10000
OPTIMIZER = 'adam'
final_acc = 0.0

#model para
batch_list = [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64]
DENSE_list = [32,64,128,256,512]
DROP_list = [1e-1,2e-1,3e-1,4e-1,5e-1]
OPTIMIZER_list = ['adam','grad','rmsp']
LR_list = [1e-6,2e-6,3e-6,4e-6,5e-6,6e-6,7e-6,8e-6,9e-6,1e-5,2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2]
epoch_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

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
                {'type': 'discrete_numeric', 'items': DENSE_list},
                {'type': 'discrete_numeric', 'items': DROP_list},
                {'type': 'discrete', 'items': OPTIMIZER_list},
                {'type': 'discrete_numeric', 'items': LR_list},
                {'type': 'discrete_numeric', 'items': epoch_list},
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
max_num_evals = 60 * 60 * 10
moo_objectives = [runtime_eval, acc_eval]
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type='realtime',config=config)
f = open("./output.log","w+")
print(pareto_opt_pts,file=f)
print("\n",file=f)
print(pareto_opt_vals,file=f)
print("\n",file=f)
print(history,file=f)