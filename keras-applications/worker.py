import os
import tensorflow as tf
import tensorflow_datasets as tfds
import nni
tfds.disable_progress_bar()


def get_default_params():
    return {
        "BATCH_SIZE": 10,
        "LEARNING_RATE": 1e-4,
        # 'MODEL': ('mnist', 48, 48, 10),
        'MODEL': ('plant_leaves', 600, 400, 22),
        'NUM_EPOCH': 2,
        "inter_op_parallelism_threads": 1,
        "intra_op_parallelism_threads": 2,
        "max_folded_constant": 6,
        "build_cost_model": 4,
        "do_common_subexpression_elimination": 1,
        "do_function_inlining": 1,
        "global_jit_level": 1,
        "infer_shapes": 1,
        "place_pruned_graph": 1,
        "enable_bfloat16_sendrecv": 1
    }


params = get_default_params()
tuned_params = nni.get_next_parameter()
params.update(tuned_params)

IMG_SIZE_LENGTH = params['MODEL'][1]
IMG_SIZE_WIDTH = params['MODEL'][2]
SHUFFLE_BUFFER_SIZE = 10
IMG_SHAPE = (IMG_SIZE_LENGTH, IMG_SIZE_WIDTH, 3)


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    # image = tf.image.grayscale_to_rgb(image)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE_LENGTH, IMG_SIZE_WIDTH))
    image = tf.cast(image, tf.float32)
    return image, label


def get_config():
    return tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=int(
            params['inter_op_parallelism_threads']),
        intra_op_parallelism_threads=int(
            params['intra_op_parallelism_threads']),
        graph_options=tf.compat.v1.GraphOptions(
            build_cost_model=int(params['build_cost_model']),
            infer_shapes=params['infer_shapes'],
            place_pruned_graph=params['place_pruned_graph'],
            enable_bfloat16_sendrecv=params['enable_bfloat16_sendrecv'],
            optimizer_options=tf.compat.v1.OptimizerOptions(
                do_common_subexpression_elimination=params['do_common_subexpression_elimination'],
                max_folded_constant_in_bytes=int(
                    params['max_folded_constant']),
                do_function_inlining=params['do_function_inlining'],
                global_jit_level=params['global_jit_level'])))


data = tfds.load(params['MODEL'][0], as_supervised=True)
train_data = data['train']
# eval_data = data['test']
train_data = train_data.map(format_example).shuffle(
    SHUFFLE_BUFFER_SIZE).batch(params['BATCH_SIZE'])
# eval_data = eval_data.map(format_example).batch(params['BATCH_SIZE'])

base_model = tf.keras.applications.ResNet50(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights=None)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(params['MODEL'][3])
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

sess = tf.compat.v1.Session(config=get_config())
tf.compat.v1.keras.backend.set_session(sess)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=params['LEARNING_RATE']),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True, reduction=tf.losses.Reduction.NONE),
              metrics=['accuracy'])

history = model.fit(train_data,
                    epochs=params['NUM_EPOCH'])
                    # validation_data=eval_data)

final_acc = history.history['accuracy'][params['NUM_EPOCH']-1]
nni.report_final_result(final_acc)
print("Final accuracy: {}".format(final_acc))
