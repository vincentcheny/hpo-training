from __future__ import absolute_import, division, print_function
from time import sleep
import grpc
import transfer_pb2
import transfer_pb2_grpc
import tensorflow as tf
import os
import json
import tensorflow_datasets as tfds
import pickle


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model


def grpc_push(trainable_var):
    channel = grpc.insecure_channel('localhost:20001')
    stub = transfer_pb2_grpc.TransferStub(channel)
    para_list = []
    for var in trainable_var:
        para_list.append(var.numpy())
    para_list = pickle.dumps(para_list)
    response = stub.UploadPara(transfer_pb2.UploadRequest(para=para_list))
    # print("push response:", response.message)


def grpc_clear(trainable_var):
    for var in trainable_var:
        var.assign(tf.zeros(shape=var.shape, dtype=var.dtype))
    # print("After grpc clear, trainable_var becomes:", trainable_var)


def grpc_pull(trainable_var):
    channel = grpc.insecure_channel('localhost:20001')
    stub = transfer_pb2_grpc.TransferStub(channel)
    response = stub.DownloadPara(transfer_pb2.DownloadRequest())
    response = pickle.loads(response.message)
    # print("pull response:", response)
    for i in range(len(trainable_var)):
        tmp = response[i]
        # print(tmp)
        trainable_var[i].assign(tf.convert_to_tensor(tmp, dtype=trainable_var[i].dtype))


# enable tensorflow-gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

tfds.disable_progress_bar()
# tf.compat.v1.disable_eager_execution()
BUFFER_SIZE = 10000
BATCH_SIZE = 32

NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = 32 * NUM_WORKERS

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:2001", "localhost:2002"]
    },
    'task': {'type': 'worker', 'index': 1}
})
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
train_datasets_unbatched = datasets['train'].map(scale).shuffle(BUFFER_SIZE)
train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)


class callbacktest(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):

        # grpc_push(multi_worker_model.trainable_variables)
        grpc_clear(multi_worker_model.trainable_variables)
        sleep(0.2)
        grpc_pull(multi_worker_model.trainable_variables)


callbacks = [callbacktest()]

with strategy.scope():
    # focus on the accuracy change.
    multi_worker_model = build_and_compile_cnn_model()
    multi_worker_model.fit(x=train_datasets,
                           epochs=9,
                           steps_per_epoch=100,
                           callbacks=callbacks)
