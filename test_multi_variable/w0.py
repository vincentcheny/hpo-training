from __future__ import absolute_import, division, print_function, \
  unicode_literals

import sys

import tensorflow as tf
import os
import json
# os.environ["SNOOPER_DISABLED"] = "0"
# import pysnooper

# tf.compat.v1.disable_eager_execution()

# from worker_model import common
tf.get_logger().setLevel('DEBUG')
BUFFER_SIZE = 10000
BATCH_SIZE = 64

workers = ["localhost:12345", "localhost:23456"]
task_index = 0

# tf.train.TFTunerContext.init_context(len(workers), task_index)

CLUSTER_SPEC = {
  'worker': workers
}
server = tf.distribute.Server(CLUSTER_SPEC, "worker", task_index)

LEARNING_RATE = 1e-4

os.environ['TF_CONFIG'] = json.dumps({
  'cluster': {
    'worker': workers
  },
  'task': {'type': 'worker', 'index': task_index}
})

# with strategy.scope():
with tf.device('/job:worker/task:0'):
  v1 = tf.Variable(10, name="v1")

with tf.compat.v1.Session(target=server.target) as sess:
  sess.run(v1.initializer)

server.join()
