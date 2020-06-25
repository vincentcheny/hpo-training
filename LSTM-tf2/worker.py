import tensorflow_datasets as tfds
import tensorflow as tf
import nni
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_default_params():
    return {
        "BATCH_SIZE":32,
        "LEARNING_RATE":1e-4,
        "DROP_OUT":5e-1,
        "DENSE_UNIT":128,
        "OPTIMIZER":"grad",
        "NUM_EPOCH":2,
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

class ReportIntermediates(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Reports intermediate accuracy to NNI framework"""
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])


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


params = get_default_params()
tuned_params = nni.get_next_parameter()
params.update(tuned_params)

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

BUFFER_SIZE = 10000
BATCH_SIZE = params['BATCH_SIZE']


train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_dataset.padded_batch(BATCH_SIZE)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(params['DENSE_UNIT'], activation='relu'),
    tf.keras.layers.Dropout(params['DROP_OUT']),
    tf.keras.layers.Dense(1)
])

if params['OPTIMIZER'] == 'adam':
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['LEARNING_RATE'])
elif params['OPTIMIZER'] == 'grad':
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=params['LEARNING_RATE'])
elif params['OPTIMIZER'] == 'rmsp':
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=params['LEARNING_RATE'])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=optimizer,
              metrics=['accuracy'])

sess = tf.compat.v1.Session(config=get_config())
tf.compat.v1.keras.backend.set_session(sess)

history = model.fit(train_dataset, epochs=params['NUM_EPOCH'],
                    validation_data=test_dataset,
                    validation_steps=20,
                    callbacks=[ReportIntermediates()])


test_loss, test_acc = model.evaluate(test_dataset)

nni.report_final_result(test_acc)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))