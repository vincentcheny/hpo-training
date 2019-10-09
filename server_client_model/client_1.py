from __future__ import absolute_import, division, print_function
from time import sleep
import tensorflow as tf
import sys
sys.path.append("..")
import grpc_op
import model_op


strategy = model_op.set_environment(index=1)
train_datasets = model_op.prepare_datasets()
run_first_time = True


class callbacktest(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        global run_first_time
        if run_first_time:
            run_first_time = False
            sleep(1)
        grpc_op.clear(multi_worker_model.trainable_variables)

        grpc_op.pull(multi_worker_model.trainable_variables)


callbacks = [callbacktest()]

with strategy.scope():
    # focus on the accuracy change.
    multi_worker_model = model_op.build_and_compile_cnn_model()
    multi_worker_model.fit(x=train_datasets,
                           epochs=9,
                           steps_per_epoch=100,
                           callbacks=callbacks)
