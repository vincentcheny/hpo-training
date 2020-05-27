import pandas as pd
import os
import numpy as np
import pandas as pd
import zipfile
import sys
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import modeling
import optimization
import run_classifier
import tokenization
import tensorflow as tf
from tqdm import tqdm
import nni
import shutil


tf.logging.set_verbosity(tf.logging.INFO)
BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = './uncased_L-12_H-768_A-12'
OUTPUT_DIR = './outputs'


train_df = pd.read_csv('./train.csv')
train_df = train_df.sample(2000)
train, test = train_test_split(train_df, test_size=0.01, random_state=42)
train_lines, train_labels = train.question_text.values, train.target.values
test_lines, test_labels = test.question_text.values, test.target.values


def create_examples(lines, set_type, labels=None):
# Generate data for the BERT model
    guid = f'{set_type}'
    examples = []
    if guid == 'train':
        for line, label in zip(lines, labels):
            text_a = line
            label = str(label)
            examples.append(
              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    else:
        for line in lines:
            text_a = line
            label = '0'
            examples.append(
              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = 32

    num_examples = len(features)

    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


def get_default_params():
    return {
        "BATCH_SIZE":32,
        "LEARNING_RATE":1e-4,
        "WARMUP_PROPORTION":0.1,
        "NUM_EPOCH":1,
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


params = get_default_params()
tuned_params = nni.get_next_parameter()
params.update(tuned_params)

# Model Hyper Parameters
TRAIN_BATCH_SIZE = params['BATCH_SIZE']
EVAL_BATCH_SIZE = 8
LEARNING_RATE = params['LEARNING_RATE']
NUM_TRAIN_EPOCHS = params['NUM_EPOCH']
WARMUP_PROPORTION = params['WARMUP_PROPORTION']
MAX_SEQ_LENGTH = 128
# Model configs
# if you wish to finetune a model on a larger dataset, use larger interval
SAVE_CHECKPOINTS_STEPS = 1000
# each checpoint weights about 1,5gb
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

label_list = ['0', '1']
tokenizer = tokenization.FullTokenizer(
    vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
train_examples = create_examples(train_lines, 'train', labels=train_labels)

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

tpu_cluster_resolver = None
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2),
    session_config=my_config)


num_train_steps = int(
    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)


model_fn = run_classifier.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=None,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=False,  # If False training will fall on CPU or GPU, depending on what is available
    use_one_hot_embeddings=True)


estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,  # If False training will fall on CPU or GPU, depending on what is available
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)


# Train the model.
print('Please wait...')
train_features = run_classifier.convert_examples_to_features(
    train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
print('***** Started training at {} *****'.format(datetime.datetime.now()))
print('  Num examples = {}'.format(len(train_examples)))
print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
tf.logging.info("  Num steps = %d", num_train_steps)
train_input_fn = run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=True)
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('***** Finished training at {} *****'.format(datetime.datetime.now()))


# eval
predict_examples = create_examples(test_lines, 'test')
predict_features = run_classifier.convert_examples_to_features(predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
predict_input_fn = input_fn_builder(
    features=predict_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)
result = estimator.predict(input_fn=predict_input_fn)
preds = []
for prediction in tqdm(result):
    for class_probability in prediction['probabilities']:
      preds.append(float(class_probability))
results = []
for i in tqdm(range(0,len(preds),2)):
  if preds[i] < 0.9:
    results.append(1)
  else:
    results.append(0)

nni.report_final_result(accuracy_score(np.array(results), test_labels))
print(f1_score(np.array(results), test_labels))
# Delete the checkpoint and summary for next trial
if os.path.exists(OUTPUT_DIR):
    try:
        shutil.rmtree(OUTPUT_DIR)
    except OSError:
        pass
