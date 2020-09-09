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

import nni
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
    x = Dropout(params['DROPOUT'])(x)
    x = Dense(params['DENSE_UNIT'], activation='relu')(x)
    x = Dropout(params['DROPOUT'])(x)
    final_output = Dense(n_out, activation='softmax', name='final_output')(x)
    model = Model(input_tensor, final_output)
    return model

def get_config():
    return tf.compat.v1.ConfigProto( 
        inter_op_parallelism_threads=int(params['inter_op_parallelism_threads']),
        intra_op_parallelism_threads=int(params['intra_op_parallelism_threads']),
        graph_options=tf.compat.v1.GraphOptions(
            infer_shapes=params['infer_shapes'],
            place_pruned_graph=params['place_pruned_graph'],
            enable_bfloat16_sendrecv=params['enable_bfloat16_sendrecv'],
            optimizer_options=tf.compat.v1.OptimizerOptions(
                do_common_subexpression_elimination=params['do_common_subexpression_elimination'],
                max_folded_constant_in_bytes=int(params['max_folded_constant']),
                do_function_inlining=params['do_function_inlining'],
                global_jit_level=params['global_jit_level'])))

# train all layers
def main():
    print(f"[INFO] Trial Config: {params}")
    start = time.time()
    batch_size = params['BATCH_SIZE'] * NUM_GPU
    train_generator = My_Generator(train_x, train_y, 128, is_train=True)
    train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=False, augment=True)
    valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)
    
    mirrored_strategy = tf.distribute.MirroredStrategy()

    if 'inter_op_parallelism_threads' in params.keys():
        sess = tf.compat.v1.Session(config=get_config())
        tf.compat.v1.keras.backend.set_session(sess)
        if params['tf_gpu_thread_mode'] in ["global", "gpu_private", "gpu_shared"]:
            os.environ['TF_GPU_THREAD_MODE'] = params['tf_gpu_thread_mode']
        cross_device_ops = params['cross_device_ops']
        num_packs = params['num_packs']
        if cross_device_ops == "HierarchicalCopyAllReduce":
            mirrored_strategy = tf.distribute.MirroredStrategy(
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=num_packs))
        elif cross_device_ops == "NcclAllReduce":
            mirrored_strategy = tf.distribute.MirroredStrategy(
                cross_device_ops=tf.distribute.NcclAllReduce(num_packs=num_packs))

    with mirrored_strategy.scope():
        is_load_model = False # args.is_load_model
        if is_load_model:
            model = tf.keras.models.load_model("../../save/inception_human_save_tpe")
        else:
            model = create_model(input_shape=(SIZE, SIZE, 3), n_out=NUM_CLASSES, mp=x)
    for layer in model.layers:
        layer.trainable = True
    
    if params['OPTIMIZER'] == 'adam':
        model.compile(loss='binary_crossentropy',
                    optimizer=Adam(lr=params['LEARNING_RATE']),
                    metrics=['accuracy', 'top_k_categorical_accuracy'])
    elif params['OPTIMIZER'] == 'sgd':
        model.compile(loss='binary_crossentropy',
                    optimizer=SGD(learning_rate=params['LEARNING_RATE']),
                    metrics=['accuracy', 'top_k_categorical_accuracy'])
    else:
        model.compile(loss='binary_crossentropy',
                    optimizer=RMSprop(learning_rate=params['LEARNING_RATE']),
                    metrics=['accuracy', 'top_k_categorical_accuracy'])

    his = model.fit(
        train_mixup,
        steps_per_epoch=np.ceil(len(train_x) / batch_size),
        validation_data=valid_generator,
        validation_steps=np.ceil(len(valid_x) / batch_size),
        epochs=epoch,
        verbose=2,
        # workers=1, use_multiprocessing=False,
        callbacks=[reduceLROnPlat])
    end = time.time()
    spent_time = (end - start) / 3600.0
    train_acc = 0. if len(his.history['accuracy']) < 1 else his.history['accuracy'][-1]
    train_top5_acc = 0. if len(his.history['top_k_categorical_accuracy']) < 1 else his.history['top_k_categorical_accuracy'][-1]
    val_acc = 0. if len(his.history['val_accuracy']) < 1 else his.history['val_accuracy'][-1]
    val_top5_acc = 0. if len(his.history['val_top_k_categorical_accuracy']) < 1 else his.history['val_top_k_categorical_accuracy'][-1]
    with open(args.log_path,"a") as f:
        print(train_acc, train_top5_acc, val_acc, val_top5_acc, abs(spent_time)*60, start, end, params,file=f)
    report_dict = {'runtime':spent_time,'default':val_acc}
    nni.report_final_result(report_dict)


def get_default_params():
    return {
	'LEARNING_RATE': 0.001,
    'OPTIMIZER':'adam',
    'BATCH_SIZE': 8,
    'NUM_EPOCH':4,
    'DENSE_UNIT': 256,
    'DROPOUT': 0.2
}

if __name__ == '__main__':
    SIZE = 200
    NUM_CLASSES = 5
    NUM_GPU = 2
    params = get_default_params()
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)
    epoch = params['TRIAL_BUDGET'] if 'TRIAL_BUDGET' in params.keys() else params['NUM_EPOCH']

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
    main()