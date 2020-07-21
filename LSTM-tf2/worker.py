import numpy as np
import random
import os
seed_value=0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)
import tensorflow_datasets as tfds
import nni


def get_default_params():
    return {
        "BATCH_SIZE":32,
        "LEARNING_RATE":1e-4,
        "DROP_OUT":5e-1,
        "DENSE_UNIT":128,
        "OPTIMIZER":"grad",
        "NUM_EPOCH":3
    }

class ReportIntermediates(tf.keras.callbacks.Callback):
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

epochs = params['NUM_EPOCH'] if 'TRIAL_BUDGET' not in params.keys() else params["TRIAL_BUDGET"]
history = model.fit(train_dataset, epochs=epochs,
                    validation_data=test_dataset,
                    validation_steps=20,
                    callbacks=[ReportIntermediates()])
print(f"history:{history}")

test_loss, test_acc = model.evaluate(test_dataset)

nni.report_final_result(test_acc)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))