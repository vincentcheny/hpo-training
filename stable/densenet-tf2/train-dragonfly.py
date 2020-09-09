import os
import time
import random
import numpy as np  # linear algebra
import pandas as pd
import warnings
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_path', type=str, default='./temp.log')
args = parser.parse_args()
print(f"seed:{args.seed}")

seed_value = args.seed
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.filterwarnings("ignore")

import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169
from tensorflow.keras.callbacks import ReduceLROnPlateau

from dragonfly import load_config, multiobjective_maximise_functions, multiobjective_minimise_functions
from dragonfly import maximise_function, minimise_function
from dragonfly.utils.option_handler import get_option_specs, load_options

from argparse import Namespace


class My_Generator(Sequence):

    def __init__(self, image_filenames, labels,
                 batch_size, is_train=True,
                 mix=False, augment=False):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.is_train = is_train
        self.is_augment = augment
        if(self.is_train):
            self.on_epoch_end()
        self.is_mix = mix

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if(self.is_train):
            return self.train_generate(batch_x, batch_y)
        return self.valid_generate(batch_x, batch_y)

    def on_epoch_end(self):
        if(self.is_train):
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
        else:
            pass

    def mix_up(self, x, y):
        lam = np.random.beta(0.2, 0.4)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        return mixed_x, mixed_y

    def train_generate(self, batch_x, batch_y):
        batch_images = []
        for (sample, label) in zip(batch_x, batch_y):
            img = cv2.imread(os.path.join(PATH, 'blindness-detection/train_images/'+sample+'.png'))
            img = cv2.resize(img, (SIZE, SIZE))
            if(self.is_augment):
                img = seq.augment_image(img)
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255
        batch_y = np.array(batch_y, np.float32)
        if(self.is_mix):
            batch_images, batch_y = self.mix_up(batch_images, batch_y)
        return batch_images, batch_y

    def valid_generate(self, batch_x, batch_y):
        batch_images = []
        for (sample, label) in zip(batch_x, batch_y):
            file_path = os.path.join(PATH, 'blindness-detection/train_images/'+sample+'.png')
            img = cv2.imread(file_path)
            img = cv2.resize(img, (SIZE, SIZE))
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255
        batch_y = np.array(batch_y, np.float32)
        return batch_images, batch_y


def create_model(input_shape, n_out, mp):
    input_tensor = Input(shape=input_shape)
    base_model = DenseNet121(include_top=False,
                             weights=None,
                             input_tensor=input_tensor)
    # base_model.load_weights("../input/densenet-keras/DenseNet-BC-121-32-no-top.h5")
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(mp[4])(x)
    x = Dense(mp[3], activation='relu')(x)
    x = Dropout(mp[4])(x)
    final_output = Dense(n_out, activation='softmax', name='final_output')(x)
    model = Model(input_tensor, final_output)
    return model

# train all layers
def runtime_eval(x):
    for i in range(len(x)):
        if type(x[i]) is np.ndarray:
            x[i] = x[i][0]
    print(f"[INFO] Trial Config: {x}")
    start = time.time()
    batch_size = x[0] * NUM_GPU
    train_generator = My_Generator(train_x, train_y, 128, is_train=True)
    train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=False, augment=True)
    valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)
    if x[15] in ["global", "gpu_private", "gpu_shared"]:
        os.environ['TF_GPU_THREAD_MODE'] = x[15]
    cross_device_ops = x[16]
    num_packs = x[17]
    if cross_device_ops == "HierarchicalCopyAllReduce":
        mirrored_strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=num_packs))
    elif cross_device_ops == "NcclAllReduce":
        mirrored_strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce(num_packs=num_packs))
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        is_load_model = False # args.is_load_model
        if is_load_model:
            model = tf.keras.models.load_model("../../save/inception_human_save_tpe")
        else:
            model = create_model(input_shape=(SIZE, SIZE, 3), n_out=NUM_CLASSES, mp=x)
    for layer in model.layers:
        layer.trainable = True
    sess =  tf.compat.v1.Session(config=tf.compat.v1.ConfigProto( 
        inter_op_parallelism_threads=x[5],
        intra_op_parallelism_threads=x[6],
        graph_options=tf.compat.v1.GraphOptions(
            infer_shapes=x[7],
            place_pruned_graph=x[8],
            enable_bfloat16_sendrecv=x[9],
            optimizer_options=tf.compat.v1.OptimizerOptions(
                do_common_subexpression_elimination=x[10],
                max_folded_constant_in_bytes=x[11],
                do_function_inlining=x[12],
                global_jit_level=x[13])),
            )
	)
    tf.compat.v1.keras.backend.set_session(sess)
    if x[14] == 'adam':
        model.compile(loss='binary_crossentropy',
                    optimizer=Adam(lr=x[1]),
                    metrics=['accuracy', 'top_k_categorical_accuracy'])
    elif x[14] == 'sgd':
        model.compile(loss='binary_crossentropy',
                    optimizer=SGD(learning_rate=x[1]),
                    metrics=['accuracy', 'top_k_categorical_accuracy'])
    else:
        model.compile(loss='binary_crossentropy',
                    optimizer=RMSprop(learning_rate=x[1]),
                    metrics=['accuracy', 'top_k_categorical_accuracy'])

    his = model.fit(
        train_mixup,
        steps_per_epoch=np.ceil(len(train_x) / batch_size),
        validation_data=valid_generator,
        validation_steps=np.ceil(len(valid_x) / batch_size),
        epochs=x[2],
        verbose=1,
        # workers=1, use_multiprocessing=False,
        callbacks=[reduceLROnPlat])
    end = time.time()
    global final_acc
    spent_time = (start - end) / 3600.0
    train_acc = 0. if len(his.history['accuracy']) < 1 else his.history['accuracy'][-1]
    train_top5_acc = 0. if len(his.history['top_k_categorical_accuracy']) < 1 else his.history['top_k_categorical_accuracy'][-1]
    val_acc = 0. if len(his.history['val_accuracy']) < 1 else his.history['val_accuracy'][-1]
    val_top5_acc = 0. if len(his.history['val_top_k_categorical_accuracy']) < 1 else his.history['val_top_k_categorical_accuracy'][-1]
    # print(his.history)
    with open(args.log_path,"a") as f:
        print(train_acc, train_top5_acc, val_acc, val_top5_acc, spent_time*(-60), start, end, x, file=f)
    final_acc = val_acc
    return spent_time


def acc_eval(x):
    global final_acc
    return float(final_acc)


SIZE = 200
NUM_CLASSES = 5
NUM_GPU = 2
final_acc = 0.0

PATH = "/uac/rshr/cyliu/bigDataStorage/dataset"
df_train = pd.read_csv(os.path.join(PATH, 'blindness-detection/train.csv'))
df_test = pd.read_csv(os.path.join(PATH, 'blindness-detection/test.csv'))

x = df_train['id_code']
y = df_train['diagnosis']

x, y = shuffle(x, y, random_state=8)

y = to_categorical(y, num_classes=NUM_CLASSES)
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,
                                                      stratify=y, random_state=8)


def sometimes(aug): return iaa.Sometimes(0.5, aug)


seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        sometimes(iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate by -20 to +20 percent (per axis)
            rotate=(-10, 10),  # rotate by -45 to +45 degrees
            shear=(-5, 5),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
                   [
            # convert images into their superpixel representation
            sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
            iaa.OneOf([
                iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(3, 5)),  # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
            ]),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),  # sharpen images
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
            # search either for all edges or for directed edges,
            # blend the result with the original image using a blobby mask
            iaa.SimplexNoiseAlpha(iaa.OneOf([
                iaa.EdgeDetect(alpha=(0.5, 1.0)),
                iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
            ])),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255),
                                      per_channel=0.5),  # add gaussian noise to images
            iaa.OneOf([
                iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove up to 10% of the pixels
                iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
            ]),
            iaa.Invert(0.01, per_channel=True),  # invert color channels
            iaa.Add((-2, 2), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
            iaa.AddToHueAndSaturation((-1, 1)),  # change hue and saturation
            # either change the brightness of the whole image (sometimes
            # per channel) or change the brightness of subareas
            iaa.OneOf([
                iaa.Multiply((0.9, 1.1), per_channel=0.5),
                iaa.FrequencyNoiseAlpha(
                    exponent=(-1, 0),
                    first=iaa.Multiply((0.9, 1.1), per_channel=True),
                    second=iaa.ContrastNormalization((0.9, 1.1))
                )
            ]),
            # move pixels locally around (with random strengths)
            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        ],
            random_order=True
        )
    ],
    random_order=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                                   verbose=1, mode='auto', epsilon=0.0001)


# model para
batch_list = [2,4,6,8,10,12,14,16]
LR_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
epoch_list = [1,2,3,4,5,6,7,8,9]
dense_list = [64,128,256,512]
dropout_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]

#hardware para
inter_list = [1,2,3,4]
intra_list = [2,4,6,8,10,12]
infer_shapes_list = [True,False]
place_pruned_graph_list = [True,False]
enable_bfloat16_sendrecv_list = [True,False]
do_common_subexpression_elimination_list = [True,False]
max_folded_constant_list = [2,4,6,8,10]
do_function_inlining_list = [True,False]
global_jit_level_list = [0,1,2]
optimizer_list = ['rmsp', 'sgd', 'adam']

#gpu para
tf_gpu_thread_mode_list = ["global", "gpu_private", "gpu_shared"]
cross_device_ops_list = ["NcclAllReduce","HierarchicalCopyAllReduce"]
num_packs_list = [0,1,2,3,4,5]

domain_vars = [
    {'type': 'discrete_numeric', 'items': batch_list},
    {'type': 'discrete_numeric', 'items': LR_list},
    {'type': 'discrete_numeric', 'items': epoch_list},
    {'type': 'discrete_numeric', 'items': dense_list},
    {'type': 'discrete_numeric', 'items': dropout_list},
    {'type': 'discrete_numeric', 'items': inter_list},
    {'type': 'discrete_numeric', 'items': intra_list},
    {'type': 'discrete', 'items': infer_shapes_list},
    {'type': 'discrete', 'items': place_pruned_graph_list},
    {'type': 'discrete', 'items': enable_bfloat16_sendrecv_list},
    {'type': 'discrete', 'items': do_common_subexpression_elimination_list},
    {'type': 'discrete_numeric', 'items': max_folded_constant_list},
    {'type': 'discrete', 'items': do_function_inlining_list},
    {'type': 'discrete_numeric', 'items': global_jit_level_list},
    {'type': 'discrete', 'items': optimizer_list},
    {'type': 'discrete', 'items': tf_gpu_thread_mode_list},
    {'type': 'discrete', 'items': cross_device_ops_list},
    {'type': 'discrete_numeric', 'items': num_packs_list}
]

# configs = [
# 	[64, 0.0005, 3,'adam'],
# 	[80, 0.0001, 2,'adam'],
# 	[8, 0.001, 5,'adam'],
# 	[8, 1e-05, 5,'adam'],
# 	[80, 5e-06, 5,'adam'],
# 	[8, 1e-05, 3,'adam'],
# 	[16, 5e-05, 4,'adam'],
# 	[80, 0.0001, 5,'adam'],
# 	[32, 0.005, 2,'adam'],
# 	[64, 0.005, 4,'adam'],
# 	[16, 0.0001, 3,'adam'],
# 	[64, 0.0005, 2,'adam']
# ]

# points = []
# for config in configs:
# 	list1, list2 = [], []
# 	for item in config:
# 		if str(item).isalpha():
# 			if isinstance(item, str):
# 				list1.insert(0, item)
# 			else:
# 				list1.append(item)
# 		else:
# 			list2.append(item)
# 	points.append([list1, list2])
# print(f'points:{points}')

# time = [
# 	[1597925277.903961,1597926447.6040773],
# 	[1597926448.5611207,1597927248.7002742],
# 	[1597927249.3406036,1597929981.4200666],
# 	[1597929982.2106106,1597932752.6701472],
# 	[1597932753.568601,1597934564.4045262],
# 	[1597934565.3196137,1597936341.5382042],
# 	[1597936342.1508777,1597938103.6379323],
# 	[1597938104.449242,1597940016.1512427],
# 	[1597940016.7480965,1597940863.9101274],
# 	[1597941102.858937,1597942715.642239],
# 	[1597942718.1823297,1597944175.3069534],
# 	[1597944179.6754997,1597945023.1937027]
# ]
# vals = [
# 	[-0.6297088861465454, 0],
# 	[-0.6207627058029175, 0],
# 	[-0.6394780874252319, 0],
# 	[-0.6291682124137878, 0],
# 	[-0.6329248547554016, 0],
# 	[-0.6395854353904724, 0],
# 	[-0.6394550800323486, 0],
# 	[-0.6329248547554016, 0],
# 	[-0.640205979347229, 0],
# 	[-0.6387096643447876, 0],
# 	[-0.6398841738700867, 0],
# 	[-0.6253763437271118, 0]]
# for i in range(len(vals)):
# 	vals[i][1] = (time[i][1]-time[i][0])/3600.0
# import copy
# true_vals= copy.deepcopy(vals)
# qinfos = []
# for i in range(len(points)):
# 	qinfo = Namespace(point=points[i], val=vals[i], true_val=true_vals[i])
# 	qinfos.append(qinfo)

# previous_eval = {'qinfos':[]}
# for i in range(len(points)):
#     tmp = Namespace(point=points[i],val=vals[i],true_val=true_vals[i])
#     previous_eval['qinfos'].append(tmp)
# p = Namespace(**previous_eval)


dragonfly_args = [
    get_option_specs('init_capital', False, 2, 'Path to the json or pb config file. '),
    get_option_specs('init_capital_frac', False, None,
                     'The fraction of the total capital to be used for initialisation.'),
    get_option_specs('num_init_evals', False, 2,
                     'The number of evaluations for initialisation. If <0, will use default.')
    # get_option_specs('prev_evaluations', False, p,'Data for any previous evaluations.'),
    # get_option_specs('prev_evaluations', False, Namespace(qinfos=qinfos),'Data for any previous evaluations.')

]

options = load_options(dragonfly_args)
config_params = {'domain': domain_vars}
config = load_config(config_params)
max_num_evals = 60*60*60
moo_objectives = [runtime_eval, acc_eval]
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(
    moo_objectives, config.domain, max_num_evals, capital_type='realtime', config=config, options=options)
f = open("./pareto-output-with-hw.log", "w+")
print(pareto_opt_pts, file=f)
print("\n", file=f)
print(pareto_opt_vals, file=f)
print("\n", file=f)
print(history, file=f)
