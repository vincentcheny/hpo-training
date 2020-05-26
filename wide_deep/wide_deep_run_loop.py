# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Core run logic for TensorFlow Wide & Deep Tutorial using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from absl import app as absl_app
from absl import flags
import tensorflow.compat.v1 as tf

import nni

LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}


def define_wide_deep_flags():
  """Add supervised learning flags, as well as wide-deep model type."""
  flags.DEFINE_string(
        name="data_dir", short_name="dd", default="/tmp",
        help="The location of the input data.")
  flags.DEFINE_string(
        name="model_dir", short_name="md", default="/tmp",
        help="The location of the model checkpoint files.")
  flags.DEFINE_boolean(
        name="clean", default=False,
        help="If set, model_dir will be removed if it exists.")
  flags.DEFINE_integer(
        name="train_epochs", short_name="te", default=1,
        help="The number of epochs used to train.")
  flags.DEFINE_integer(
        name="epochs_between_evals", short_name="ebe", default=1,
        help="The number of training epochs to run between "
              "evaluations.")
  flags.DEFINE_float(
        name="stop_threshold", short_name="st",
        default=None,
        help="If passed, training will stop at the earlier of "
              "train_epochs and when the evaluation metric is  "
              "greater than or equal to stop_threshold.")
  flags.DEFINE_integer(
        name="batch_size", short_name="bs", default=32,
        help="Batch size for training and evaluation")
  flags.DEFINE_integer(
        name="num_gpus", short_name="ng",
        default=1,
        help="How many GPUs to use at each worker with the "
            "DistributionStrategies API. The default is 1.")
  flags.DEFINE_boolean(
        name="run_eagerly", default=False,
        help="Run the model op by op without building a model function.")
  flags.DEFINE_string(
        name="distribution_strategy", short_name="ds", default="mirrored",
        help="The Distribution Strategy to use for training. "
    )
  flags.DEFINE_list(
        name="hooks", short_name="hk", default="LoggingTensorHook", help="")
  flags.DEFINE_string(
        name="export_dir", short_name="ed", default=None, help="")

  flags.DEFINE_enum(
      name="model_type", short_name="mt", default="wide_deep",
      enum_values=['wide', 'deep', 'wide_deep'],
      help="Select model topology.")
  flags.DEFINE_boolean(
      name="download_if_missing", default=True, 
      help="Download data to data_dir if it is not already present.")


def export_model(model, model_type, export_dir, model_column_fn):
  """Export to SavedModel format.

  Args:
    model: Estimator object
    model_type: string indicating model type. "wide", "deep" or "wide_deep"
    export_dir: directory to export the model.
    model_column_fn: Function to generate model feature columns.
  """
  wide_columns, deep_columns = model_column_fn()
  if model_type == 'wide':
    columns = wide_columns
  elif model_type == 'deep':
    columns = deep_columns
  else:
    columns = wide_columns + deep_columns
  feature_spec = tf.feature_column.make_parse_example_spec(columns)
  example_input_fn = (
      tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
  model.export_savedmodel(export_dir, example_input_fn,
                          strip_default_attrs=True)


def run_loop(name, train_input_fn, eval_input_fn, model_column_fn,
             build_estimator_fn, flags_obj, tensors_to_log, early_stop=False):
  """Define training loop."""
  model = build_estimator_fn(
      model_dir=flags_obj.model_dir, model_type=flags_obj.model_type,
      model_column_fn=model_column_fn)


  loss_prefix = LOSS_PREFIX.get(flags_obj.model_type, '')
  tensors_to_log = {k: v.format(loss_prefix=loss_prefix)
                    for k, v in tensors_to_log.items()}

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  for n in range(flags_obj.train_epochs // flags_obj.epochs_between_evals):
    model.train(input_fn=train_input_fn)

    results = model.evaluate(input_fn=eval_input_fn)

    # Display evaluation metrics
    tf.logging.info('Results at epoch %d / %d',
                    (n + 1) * flags_obj.epochs_between_evals,
                    flags_obj.train_epochs)
    tf.logging.info('-' * 60)

    for key in sorted(results):
      tf.logging.info('%s: %s' % (key, results[key]))

    nni.report_intermediate_result(results['accuracy'])
  
  nni.report_final_result(results['accuracy'])
