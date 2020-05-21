import tensorflow_datasets as tfds
import tensorflow as tf
import nni

def get_default_params():
    return {
        "BATCH_SIZE":32,
        "LEARNING_RATE":1e-4,
        "DROP_OUT":5e-1,
        "DENSE_UNIT":128,
        "OPTIMIZER":"grad",
        "KERNEL_SIZE":3,
        "NUM_EPOCH":6,
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

class ReportIntermediates(Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Reports intermediate accuracy to NNI framework"""
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])


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
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=params['NUM_EPOCH'],
                    validation_data=test_dataset,
                    validation_steps=30,
                    callbacks=[ReportIntermediates()])


test_loss, test_acc = model.evaluate(test_dataset)

nni.report_final_result(test_acc)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))