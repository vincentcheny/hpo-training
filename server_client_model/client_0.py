from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import datetime
import sys

sys.path.append("..")
import grpc_op
import model_op


strategy = model_op.set_environment(index=0)
train_datasets = model_op.prepare_datasets()


class callbacktest(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}

    def on_batch_end(self, batch, logs=None):
        # Record batch loss on tensorboard
        # if logs is None:
        #     logs = {}
        # tf.summary.scalar('batch loss', data=logs.get('loss'), step=batch)
        # print(" - logs.loss:", logs.get('loss'))

        grpc_op.push(multi_worker_model.trainable_variables)


log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
file_writer = tf.summary.create_file_writer(log_dir + "\\metrics")
file_writer.set_as_default()

checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

callbacks = [callbacktest(), tensorboard_callback]

with strategy.scope():
    # focus on the accuracy change.
    multi_worker_model = model_op.build_and_compile_cnn_model()
    multi_worker_model.fit(x=train_datasets,
                           epochs=9,
                           steps_per_epoch=100,
                           callbacks=callbacks)
