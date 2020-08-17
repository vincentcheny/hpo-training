seed_value=0
import random
random.seed(seed_value)
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
import numpy as np
np.random.seed(seed_value)
import keras
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)
from functools import partial
import time
import nni
from tensorflow.python.client import device_lib
from googlenet import GoogLeNetBN



def resize_and_rescale_image(image, height, width,do_mean_subtraction=True, scope=None):
    with tf.compat.v1.name_scope(values=[image, height, width], name=scope,
                       default_name='resize_image'):
        image = tf.expand_dims(image, 0)
        image = tf.compat.v1.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
        if do_mean_subtraction:
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
    return image


def decode_jpeg(image_buffer, scope=None):
    with tf.compat.v1.name_scope(values=[image_buffer], name=scope,default_name='decode_jpeg'):
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def _parse_fn(example_serialized, is_training):
    feature_map = {
        'image/encoded': tf.compat.v1.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.compat.v1.FixedLenFeature([], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.compat.v1.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    parsed = tf.compat.v1.parse_single_example(example_serialized, feature_map)
    image = decode_jpeg(parsed['image/encoded'])
    image = resize_and_rescale_image(image, 224, 224)
    label = tf.one_hot(parsed['image/class/label'] - 1, NUM_CLASSES, dtype=tf.float32)
    return (image, label)



def get_dataset(tfrecords_dir, subset, batch_size):
    files = tf.compat.v1.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64),seed=0)
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=8192,seed=0)
    parser = partial(
        _parse_fn, is_training=True if subset == 'train' else False)
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=parser,
            batch_size=batch_size,
            num_parallel_calls=NUM_DATA_WORKERS))
    dataset = dataset.prefetch(batch_size)
    return dataset


def _set_l2(model, weight_decay):
    for layer in model.layers:
        # if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        #     layer.add_loss(lambda: tf.keras.regularizers.l2(weight_decay)(layer.depthwise_kernel))
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.add_loss(lambda: tf.keras.regularizers.l2(weight_decay)(layer.kernel))
        elif isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l2(weight_decay)(layer.kernel))


def get_lr_func(total_epochs, lr_sched='exp',initial_lr=1e-2, final_lr=1e-5, NUM_GPU=1):
    def linear_decay(epoch):
        """Decay LR linearly for each epoch."""
        if total_epochs == 1:
            return initial_lr
        else:
            ratio = max((total_epochs - epoch - 1.) / (total_epochs - 1.), 0.)
            lr = final_lr + (initial_lr - final_lr) * ratio
            print('Epoch %d, lr = %f' % (epoch+1, lr))
            return lr

    def exp_decay(epoch):
        """Decay LR exponentially for each epoch."""
        if total_epochs == 1:
            return initial_lr
        else:
            lr_decay = (final_lr / initial_lr) ** (1. / (total_epochs - 1))
            lr = initial_lr * (lr_decay ** epoch)
            print('Epoch %d, lr = %f' % (epoch+1, lr))
            return lr

    if total_epochs < 1:
        raise ValueError('bad total_epochs (%d)' % total_epochs)
    if lr_sched == 'linear':
        return tf.keras.callbacks.LearningRateScheduler(linear_decay)
    elif lr_sched == 'exp':
        return tf.keras.callbacks.LearningRateScheduler(exp_decay)
    else:
        raise ValueError('bad lr_sched')


def get_optimizer(optim_name, initial_lr, epsilon=1e-2):
    if optim_name == 'sgd':
        return tf.keras.optimizers.SGD(lr=initial_lr, momentum=0.9)
    elif optim_name == 'adam':
        return tf.keras.optimizers.Adam(lr=initial_lr, epsilon=epsilon)
    elif optim_name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(lr=initial_lr, epsilon=epsilon,
                                           rho=0.9)
    else:
        raise ValueError


def config_keras_backend(params=None):
    print(f"\nHardware params:{params}")
    """Config tensorflow backend to use less GPU memory."""
    if not params:
        print("\nUse default hardware config.")
        config = tf.compat.v1.ConfigProto()
    else:
        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=int(params["inter_op_parallelism_threads"]),
            intra_op_parallelism_threads=int(params["intra_op_parallelism_threads"]),
            graph_options=tf.compat.v1.GraphOptions(
                infer_shapes=params["infer_shapes"],
                place_pruned_graph=params["place_pruned_graph"],
                enable_bfloat16_sendrecv=params["enable_bfloat16_sendrecv"],
                optimizer_options=tf.compat.v1.OptimizerOptions(
                    do_common_subexpression_elimination=params["do_common_subexpression_elimination"],
                    max_folded_constant_in_bytes=params["max_folded_constant"],
                    do_function_inlining=params["do_function_inlining"],
                    global_jit_level=params["global_jit_level"])))
        print("\nUse customized hardware config.")
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)


def clear_keras_session():
    tf.compat.v1.keras.backend.clear_session()



def get_default_params():
    return {
      "EPSILON":0.1,
      "BATCH_SIZE":32,
      "OPTIMIZER":"adam",
      "INIT_LR":1e-2,
      "FINAL_LR":1e-6,
      "WEIGHT_DECAY":2e-4,
      "NUM_EPOCH":1,
      "inter_op_parallelism_threads":1,
      "intra_op_parallelism_threads":1,
      "max_folded_constant":2,
      "do_common_subexpression_elimination":0,
      "do_function_inlining":0,
      "global_jit_level":0,
      "infer_shapes":0,
      "place_pruned_graph":0,
      "enable_bfloat16_sendrecv":0,
      "PREFERENCE":"accuracy",
      "cross_device_ops_list":"NcclAllReduce",
      "num_packs_list":2,
      "tf_gpu_thread_mode_list":"gpu_shared"
    }


params = get_default_params()
tuned_params = nni.get_next_parameter()
params.update(tuned_params)


## global fix variable 
dataset_dir = '/uac/rshr/cyliu/bigDataStorage/imagenet_500classes/'
NUM_CLASSES = 500
IN_SHAPE = (224, 224, 3)
NUM_DATA_WORKERS = 12
# NUM_GPU = 1
NUM_GPU = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
print(f"Using {NUM_GPU} GPUs to train.")
NUM_DISTRIBUTE = NUM_GPU if NUM_GPU > 0 else 1

## seachspace & adjusted variable
if params["tf_gpu_thread_mode_list"] in ["global", "gpu_private", "gpu_shared"]:
    os.environ['TF_GPU_THREAD_MODE'] = params["tf_gpu_thread_mode_list"]
epochs_to_run = params["NUM_EPOCH"] if 'TRIAL_BUDGET' not in params.keys() else params["TRIAL_BUDGET"]
epsilon = params["EPSILON"]
batch_size = params["BATCH_SIZE"]
optimizer_name = params["OPTIMIZER"]
init_lr = params["INIT_LR"]
final_lr = params["FINAL_LR"]
weight_decay = params["WEIGHT_DECAY"]
train_steps = int(642289 / batch_size) #for 500 class
val_steps = int(25000 / batch_size) #for 500 calss


ds_train = get_dataset(dataset_dir, 'train', batch_size)
ds_valid = get_dataset(dataset_dir, 'validation', batch_size)


## BUILD MODEL
cross_device_ops = params["cross_device_ops_list"]
num_packs = params["num_packs_list"]
if cross_device_ops == "HierarchicalCopyAllReduce":
    mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=num_packs))
elif cross_device_ops == "NcclAllReduce":
    mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce(num_packs=num_packs))
else:
    mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    # base = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IN_SHAPE, include_top=False, weights=None, classes=NUM_CLASSES) ## this is replaceable, reference to keras_imagenet/models/models.py
    # base = GoogLeNetBN(input_shape=IN_SHAPE, include_top=False, weights=None, classes=NUM_CLASSES) ## this is replaceable, reference to keras_imagenet/models/models.py
    # x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    # kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    # bias_initializer = tf.constant_initializer(value=0.0)
    # x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='Logits',kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)(x)
    # model = tf.keras.models.Model(inputs=base.input, outputs=x)
    model = tf.keras.models.load_model("../save/imagenet_save")
    _set_l2(model, weight_decay)
    smooth = 0.1
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=smooth)
    # loss = 'categorical_crossentropy'


    for layer in model.layers:
        layer.trainable = True


    opt = get_optimizer(optimizer_name,init_lr,epsilon)
    model.compile(optimizer=opt,loss=loss,metrics=['accuracy', 'top_k_categorical_accuracy'])


config_keras_backend(params)
start = time.time()
print(f"[INFO] Total Epochs:{epochs_to_run} Train Steps:{train_steps} Validate Steps: {val_steps} Workers:{NUM_DISTRIBUTE} Batch size:{batch_size}")
his = model.fit(
    x=ds_train,
    steps_per_epoch=train_steps,
    validation_data=ds_valid,
    validation_steps=val_steps,
    epochs=epochs_to_run,
    verbose=2,
    callbacks=[get_lr_func(total_epochs=epochs_to_run, initial_lr=init_lr, final_lr=final_lr, NUM_GPU=NUM_GPU)]
    )
end = time.time()
clear_keras_session()

spent_time = (start - end) / 3600.0
final_acc = 0. if len(his.history['val_top_k_categorical_accuracy']) < 1 else his.history['val_top_k_categorical_accuracy'][-1]

report_dict = {'accuracy':final_acc,'runtime':spent_time,'default':final_acc}
nni.report_final_result(report_dict)
print(f"Final acc:{final_acc}")  
