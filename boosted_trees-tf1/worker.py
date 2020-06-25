from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from absl import app as absl_app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
import nni


NPZ_FILE = "HIGGS.csv.gz.npz"  # numpy compressed file containing "data" array


def read_higgs_data(data_dir, train_start, train_count, eval_start, eval_count):
  """Reads higgs data from csv and returns train and eval data.

  Args:
    data_dir: A string, the directory of higgs dataset.
    train_start: An integer, the start index of train examples within the data.
    train_count: An integer, the number of train examples within the data.
    eval_start: An integer, the start index of eval examples within the data.
    eval_count: An integer, the number of eval examples within the data.

  Returns:
    Numpy array of train data and eval data.
  """
  npz_filename = os.path.join(data_dir, NPZ_FILE)
  try:
    # gfile allows numpy to read data from network data sources as well.
    with tf.gfile.Open(npz_filename, "rb") as npz_file:
      with np.load(npz_file) as npz:
        data = npz["data"]
  except tf.errors.NotFoundError as e:
    raise RuntimeError(
        "Error loading data; use data_download.py to prepare the data.\n{}: {}"
        .format(type(e).__name__, e))
  return (data[train_start:train_start+train_count],
          data[eval_start:eval_start+eval_count])


# This showcases how to make input_fn when the input data is available in the
# form of numpy arrays.
def make_inputs_from_np_arrays(features_np, label_np):
  """Makes and returns input_fn and feature_columns from numpy arrays.

  The generated input_fn will return tf.data.Dataset of feature dictionary and a
  label, and feature_columns will consist of the list of
  tf.feature_column.BucketizedColumn.

  Note, for in-memory training, tf.data.Dataset should contain the whole data
  as a single tensor. Don't use batch.

  Args:
    features_np: A numpy ndarray (shape=[batch_size, num_features]) for
        float32 features.
    label_np: A numpy ndarray (shape=[batch_size, 1]) for labels.

  Returns:
    input_fn: A function returning a Dataset of feature dict and label.
    feature_names: A list of feature names.
    feature_column: A list of tf.feature_column.BucketizedColumn.
  """
  num_features = features_np.shape[1]
  features_np_list = np.split(features_np, num_features, axis=1)
  # 1-based feature names.
  feature_names = ["feature_%02d" % (i + 1) for i in range(num_features)]

  # Create source feature_columns and bucketized_columns.
  def get_bucket_boundaries(feature):
    """Returns bucket boundaries for feature by percentiles."""
    return np.unique(np.percentile(feature, range(0, 100))).tolist()
  source_columns = [
      tf.feature_column.numeric_column(
          feature_name, dtype=tf.float32,
          # Although higgs data have no missing values, in general, default
          # could be set as 0 or some reasonable value for missing values.
          default_value=0.0)
      for feature_name in feature_names
  ]
  bucketized_columns = [
      tf.feature_column.bucketized_column(
          source_columns[i],
          boundaries=get_bucket_boundaries(features_np_list[i]))
      for i in range(num_features)
  ]

  # Make an input_fn that extracts source features.
  def input_fn():
    """Returns features as a dictionary of numpy arrays, and a label."""
    features = {
        feature_name: tf.constant(features_np_list[i])
        for i, feature_name in enumerate(feature_names)
    }
    return tf.data.Dataset.zip((tf.data.Dataset.from_tensors(features),
                                tf.data.Dataset.from_tensors(label_np),))

  return input_fn, feature_names, bucketized_columns


def make_eval_inputs_from_np_arrays(features_np, label_np):
  """Makes eval input as streaming batches."""
  num_features = features_np.shape[1]
  features_np_list = np.split(features_np, num_features, axis=1)
  # 1-based feature names.
  feature_names = ["feature_%02d" % (i + 1) for i in range(num_features)]

  def input_fn():
    features = {
        feature_name: tf.constant(features_np_list[i])
        for i, feature_name in enumerate(feature_names)
    }
    return tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(features),
        tf.data.Dataset.from_tensor_slices(label_np),)).batch(1000)

  return input_fn


def _make_csv_serving_input_receiver_fn(column_names, column_defaults):
  """Returns serving_input_receiver_fn for csv.

  The input arguments are relevant to `tf.decode_csv()`.

  Args:
    column_names: a list of column names in the order within input csv.
    column_defaults: a list of default values with the same size of
        column_names. Each entity must be either a list of one scalar, or an
        empty list to denote the corresponding column is required.
        e.g. [[""], [2.5], []] indicates the third column is required while
            the first column must be string and the second must be float/double.

  Returns:
    a serving_input_receiver_fn that handles csv for serving.
  """
  def serving_input_receiver_fn():
    csv = tf.placeholder(dtype=tf.string, shape=[None], name="csv")
    features = dict(zip(column_names, tf.decode_csv(csv, column_defaults)))
    receiver_tensors = {"inputs": csv}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  return serving_input_receiver_fn


def train_boosted_trees(flags_obj):
  """Train boosted_trees estimator on HIGGS data.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  # Download data if not present
  data_dir = flags_obj.data_dir
  if not os.path.isdir(data_dir):
    print(f"No data found. Start downloading {os.path.realpath(data_dir)}/{NPZ_FILE}...")
    os.system("python data_download.py")
  # Clean up the model directory if present.
  if tf.gfile.Exists(flags_obj.model_dir):
    tf.gfile.DeleteRecursively(flags_obj.model_dir)
  tf.logging.info("## Data loading...")
  train_data, eval_data = read_higgs_data(
      flags_obj.data_dir, flags_obj.train_start, flags_obj.train_count,
      flags_obj.eval_start, flags_obj.eval_count)
  tf.logging.info("## Data loaded; train: {}{}, eval: {}{}".format(
      train_data.dtype, train_data.shape, eval_data.dtype, eval_data.shape))
  # Data consists of one label column followed by 28 feature columns.
  train_input_fn, feature_names, feature_columns = make_inputs_from_np_arrays(
      features_np=train_data[:, 1:], label_np=train_data[:, 0:1])
  eval_input_fn = make_eval_inputs_from_np_arrays(
      features_np=eval_data[:, 1:], label_np=eval_data[:, 0:1])
  tf.logging.info("## Features prepared. Training starts...")

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
  # Though BoostedTreesClassifier is under tf.estimator, faster in-memory
  # training is yet provided as a contrib library.
  from tensorflow.contrib import estimator as contrib_estimator  # pylint: disable=g-import-not-at-top
  classifier = contrib_estimator.boosted_trees_classifier_train_in_memory(
      train_input_fn,
      feature_columns,
      model_dir=flags_obj.model_dir or None,
      n_trees=flags_obj.n_trees,
      max_depth=flags_obj.max_depth,
      learning_rate=flags_obj.learning_rate,
      config=tf.estimator.RunConfig(session_config=my_config))

  # Evaluation.
  eval_results = classifier.evaluate(eval_input_fn)
  nni.report_final_result(eval_results['accuracy'])


def main(_):
  train_boosted_trees(flags.FLAGS)


def define_train_higgs_flags():
  """Add tree related flags as well as training/eval configuration."""

  flags.DEFINE_string(
        name="data_dir", short_name="dd", default="./higgs_data",
        help="The location of the input data.")
  flags.DEFINE_string(
        name="model_dir", short_name="md", default="./higgs_ckpt",
        help="The location of the model checkpoint files.")
  flags.DEFINE_boolean(
        name="clean", default=True,
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
        name="num_gpus", short_name="ng",
        default=1,
        help=
            "How many GPUs to use at each worker with the "
            "DistributionStrategies API. The default is 1.")
  flags.DEFINE_list(
        name="hooks", short_name="hk", default="LoggingTensorHook",
        help=
            u"A list of (case insensitive) strings to specify the names of "
            u"training hooks. Example: `--hooks ProfilerHook,"
            u"ExamplesPerSecondHook`\n See hooks_helper "
            u"for details.")
  flags.DEFINE_string(
        name="export_dir", short_name="ed", default=None,
        help="If set, a SavedModel serialization of the model will "
                       "be exported to this directory at the end of training. "
                       "See the README for more details and relevant links.")

  flags.DEFINE_integer(
      name="train_start", default=0,
      help="Start index of train examples within the data.")
  flags.DEFINE_integer(
      name="train_count", default=1000000,
      help="Number of train examples within the data.")
  flags.DEFINE_integer(
      name="eval_start", default=10000000,
      help="Start index of eval examples within the data.")
  flags.DEFINE_integer(
      name="eval_count", default=1000000,
      help="Number of eval examples within the data.")

  flags.DEFINE_integer(
      "n_trees", default=100, help="Number of trees to build.")
  flags.DEFINE_integer(
      "max_depth", default=6, help="Maximum depths of each tree.")
  flags.DEFINE_float(
      "learning_rate", default=0.1,
      help="The learning rate.")

def get_default_params():
    return {
        "N_TREES":100,
        "LEARNING_RATE":1e-1,
        "MAX_DEPTH":6,
        "NUM_EXAMPLES":1e6,
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

if __name__ == "__main__":
  # Training progress and eval results are shown as logging.INFO; so enables it.
  tf.logging.set_verbosity(tf.logging.INFO)
  define_train_higgs_flags()
  
  params = get_default_params()
  received_params = nni.get_next_parameter()
  params.update(received_params)
  flags.FLAGS.train_count = int(params["NUM_EXAMPLES"])
  flags.FLAGS.n_trees = params["N_TREES"]
  flags.FLAGS.max_depth = params["MAX_DEPTH"]
  flags.FLAGS.learning_rate = params["LEARNING_RATE"]

  absl_app.run(main)
