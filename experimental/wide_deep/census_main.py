# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train DNN on census income dataset."""

import os

from absl import app as absl_app
from absl import flags
import tensorflow.compat.v1 as tf
import census_dataset
import wide_deep_run_loop
import nni
import shutil


def set_defaults(**kwargs):
  for key, value in kwargs.items():
    flags.FLAGS.set_default(name=key, value=value)

def define_census_flags():
  wide_deep_run_loop.define_wide_deep_flags()
  flags.adopt_module_key_flags(wide_deep_run_loop)
  set_defaults(data_dir='/uac/rshr/cyliu/bigDataStorage/moo/chen.yu/census_data',
              model_dir='/uac/rshr/cyliu/bigDataStorage/moo/chen.yu/census_model',
  # set_defaults(data_dir='./census_data',
  #             model_dir='./census_model',
              train_epochs=10,
              epochs_between_evals=10,
              batch_size=40)


def build_estimator(model_dir, model_type, model_column_fn):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = model_column_fn()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  my_config = tf.ConfigProto( 
    inter_op_parallelism_threads=int(params['inter_op_parallelism_threads']),
    intra_op_parallelism_threads=int(params['intra_op_parallelism_threads']),
    graph_options=tf.GraphOptions(
        build_cost_model=int(params['build_cost_model']),
        infer_shapes=params['infer_shapes'],
        place_pruned_graph=params['place_pruned_graph'],
        enable_bfloat16_sendrecv=params['enable_bfloat16_sendrecv'],
        optimizer_options=tf.OptimizerOptions(
            do_common_subexpression_elimination=params['do_common_subexpression_elimination'],
            max_folded_constant_in_bytes=int(params['max_folded_constant']),
            do_function_inlining=params['do_function_inlining'],
            global_jit_level=params['global_jit_level'])))
  run_config = tf.estimator.RunConfig(session_config=my_config)
  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config,
        optimizer=tf.keras.optimizers.Ftrl(learning_rate=params['LEARNING_RATE']))
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config,
        optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=params['LEARNING_RATE']))
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
        dnn_optimizer=tf.train.AdamOptimizer(learning_rate=params['LEARNING_RATE']))


def run_census(flags_obj):
  """Construct all necessary functions and call run_loop.

  Args:
    flags_obj: Object containing user specified flags.
  """
  if flags_obj.download_if_missing:
    census_dataset.download(flags_obj.data_dir)

  train_file = os.path.join(flags_obj.data_dir, census_dataset.TRAINING_FILE)
  test_file = os.path.join(flags_obj.data_dir, census_dataset.EVAL_FILE)

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return census_dataset.input_fn(
        train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)

  def eval_input_fn():
    return census_dataset.input_fn(test_file, 1, False, flags_obj.batch_size)

  tensors_to_log = {
      'average_loss': '{loss_prefix}head/truediv',
      'loss': '{loss_prefix}head/weighted_loss/Sum'
  }

  wide_deep_run_loop.run_loop(
      name="Census Income", train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      model_column_fn=census_dataset.build_model_columns,
      build_estimator_fn=build_estimator,
      flags_obj=flags_obj,
      tensors_to_log=tensors_to_log,
      early_stop=True)
  if os.path.exists(flags_obj.model_dir):
    shutil.rmtree(flags_obj.model_dir)


def main(_):
  run_census(flags.FLAGS)

def get_default_params():
  return {
    "BATCH_SIZE":32,
    "LEARNING_RATE":1e-4,
    "NUM_EPOCHS":30,
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

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_census_flags()
  params = get_default_params()
  received_params = nni.get_next_parameter()
  params.update(received_params)
  flags.FLAGS.train_epochs = params["NUM_EPOCHS"]
  flags.FLAGS.batch_size = params['BATCH_SIZE']
  absl_app.run(main)
