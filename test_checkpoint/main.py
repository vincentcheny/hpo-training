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
task_index = int(sys.argv[1])

# tf.train.TFTunerContext.init_context(len(workers), task_index)

CLUSTER_SPEC = {
  'worker': workers
}
# tf.distribute.Server(CLUSTER_SPEC, "worker", task_index)

LEARNING_RATE = 1e-4

os.environ['TF_CONFIG'] = json.dumps({
  'cluster': {
    'worker': workers
  },
  'task': {'type': 'worker', 'index': task_index}
})


def getClusterAndServerDef():
  tf_config_json = os.environ.get("TF_CONFIG", "{}")
  tf_config = json.loads(tf_config_json)

  task = tf_config.get("task", {})
  cluster_spec = tf_config.get("cluster", {})
  cluster_spec_object = tf.train.ClusterSpec(cluster_spec)
  job_name = task["type"]
  task_id = task["index"]
  server_def = tf.train.ServerDef(
      cluster=cluster_spec_object.as_cluster_def(),
      protocol="grpc",
      job_name=job_name,
      task_index=task_id)
  return cluster_spec_object, server_def

cluster_spec, server_def = getClusterAndServerDef()

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# tf.config.experimental_connect_to_cluster(cluster_spec,
#                                           "worker",
#                                           task_index)

# context:tf.distribute.ReplicaContext = tf.distribute.get_replica_context()
# context2 = tf.distribute.get_strategy()
# tf.config.experimental_connect_to_cluster()

# with strategy.scope():
with tf.device('/job:worker/task:1'):
  v1 = tf.Variable(1, name="v1")
  v2 = tf.Variable(2, name="v2")

print(v1.read_value())
