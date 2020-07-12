import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
from dragonfly import load_config, multiobjective_maximise_functions,multiobjective_minimise_functions
from dragonfly import maximise_function,minimise_function
import time
import shutil


def create_examples(lines, set_type, labels=None):
#Generate data for the BERT model
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
    print(params)
    global BS
    batch_size = BS

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



def runtime_eval(x):
    # Model Hyper Parameters
    global BS
    BS = x[0]
    TRAIN_BATCH_SIZE = x[0]
    NUM_TRAIN_EPOCHS = x[1]
    LEARNING_RATE = x[2]
    WARMUP_PROPORTION = x[3]
    EVAL_BATCH_SIZE = 8
    MAX_SEQ_LENGTH = 128
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 100000000000000 #if you wish to finetune a model on a larger dataset, use larger interval
    # each checpoint weights about 1,5gb
    ITERATIONS_PER_LOOP = 1000
    NUM_TPU_CORES = 8
    VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
    CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
    INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

    label_list = ['0', '1']
    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
    train_examples = create_examples(train_lines, 'train', labels=train_labels)

    tpu_cluster_resolver = None
    my_config = tf.ConfigProto( 
        inter_op_parallelism_threads=x[4],
        intra_op_parallelism_threads=x[5],
        graph_options=tf.GraphOptions(
            build_cost_model=x[6],
            infer_shapes=x[7],
            place_pruned_graph=x[8],
            enable_bfloat16_sendrecv=x[9],
            optimizer_options=tf.OptimizerOptions(
                do_common_subexpression_elimination=x[10],
                max_folded_constant_in_bytes=x[11],
                do_function_inlining=x[12],
                global_jit_level=x[13])))
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=OUTPUT_DIR,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        session_config=my_config,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=ITERATIONS_PER_LOOP,
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))


    num_train_steps = int(len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)


    model_fn = run_classifier.model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
        num_labels=len(label_list),
        init_checkpoint=None,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available  
        use_one_hot_embeddings=True)


    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available 
        model_fn=model_fn,
        config=run_config,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE)

    #prepare for eval
    predict_examples = create_examples(test_lines, 'test')
    predict_features = run_classifier.convert_examples_to_features(predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = input_fn_builder(
        features=predict_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)


    # Train the model.
    tf.logging.set_verbosity(tf.logging.INFO)
    print('Please wait...')
    train_features = run_classifier.convert_examples_to_features(
        train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    print('***** Started training at {} *****'.format(datetime.datetime.now()))
    print('  Num examples = {}'.format(len(train_examples)))
    print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
    tf.logging.info("  Num steps = %d", num_train_steps)
    

    ##start counting time of training
    start = time.time()
    train_input_fn = run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print('***** Finished training at {} *****'.format(datetime.datetime.now()))



    ##eval
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

    ##end of train and eval
    end = time.time()
    print(accuracy_score(np.array(results), test_labels))
    print(f1_score(np.array(results), test_labels))
    global final_acc
    final_acc = accuracy_score(np.array(results), test_labels)
    global spent_time
    spent_time = end - start
    return -float(spent_time)


def acc_eval(x):
    try:
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR,ignore_errors=True)
            print("clean is finish")
    except OSError:
        print("OS error")
    global final_acc
    return final_acc


BS = 16
final_acc = 0.0
spent_time = 0.0
BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = './uncased_L-12_H-768_A-12'
OUTPUT_DIR = './outputs'


train_df =  pd.read_csv('./train.csv')
train_df = train_df.sample(3000)
train, test = train_test_split(train_df, test_size = 0.1, random_state=42)
train_lines, train_labels = train.question_text.values, train.target.values
test_lines, test_labels = test.question_text.values, test.target.values


#model para
batch_list = [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64]
epoch_list = [1,2,3]
LR_list = [1e-6,2e-6,3e-6,4e-6,5e-6,6e-6,7e-6,8e-6,9e-6,1e-5,2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2]
warmup_list = [0.05,0.1,0.15]

#hardware para
inter_list = [1,2,3,4]
intra_list = [2,4,6,8,10,12]
build_cost_model_list = [0,2,4,6,8]
infer_shapes_list = [True,False]
place_pruned_graph_list = [True,False]
enable_bfloat16_sendrecv_list = [True,False]
do_common_subexpression_elimination_list = [True,False]
max_folded_constant_list = [2,4,6,8,10]
do_function_inlining_list = [True,False]
global_jit_level_list = [0,1,2]

domain_vars = [{'type': 'discrete_numeric', 'items': batch_list},
              {'type': 'discrete_numeric', 'items': epoch_list},
              {'type': 'discrete_numeric', 'items': LR_list},
              {'type': 'discrete_numeric', 'items': warmup_list},
              {'type': 'discrete_numeric', 'items': inter_list},
              {'type': 'discrete_numeric', 'items': intra_list},
              {'type': 'discrete_numeric', 'items': build_cost_model_list},
              {'type': 'discrete', 'items': infer_shapes_list},
              {'type': 'discrete', 'items': place_pruned_graph_list},
              {'type': 'discrete', 'items': enable_bfloat16_sendrecv_list},
              {'type': 'discrete', 'items': do_common_subexpression_elimination_list},
              {'type': 'discrete_numeric', 'items': max_folded_constant_list},
              {'type': 'discrete', 'items': do_function_inlining_list},
              {'type': 'discrete_numeric', 'items': global_jit_level_list}
            ]
config_params = {'domain': domain_vars}
config = load_config(config_params)
max_num_evals = 60 * 60 * 10
moo_objectives = [runtime_eval, acc_eval]
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type='realtime',config=config)
f = open("./output.log","w+")
print(pareto_opt_pts,file=f)
print("\n",file=f)
print(pareto_opt_vals,file=f)
print("\n",file=f)
print(history,file=f)
