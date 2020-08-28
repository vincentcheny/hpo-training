import sys
import os
import time
import nni
import argparse
import numpy as np
import random
import pandas as pd
import skimage.io
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image
import cv2
import warnings
from sklearn.utils import class_weight, shuffle

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--kernel_initializer', type=str, default='glorot_uniform')
parser.add_argument('--is_load_model', type=int, default=0)
parser.add_argument('--is_augument', type=int, default=1)
parser.add_argument('--log_path', type=str, default='./temp.log')
parser.add_argument('--is_soo', type=int, default=1)

args = parser.parse_args()
print(f"seed:{args.seed} kernel_initializer:{args.kernel_initializer} is_load_model:{args.is_load_model} is_augument:{args.is_augument}")

seed_value = args.seed
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
np.random.seed(seed_value)
random.seed(seed_value)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split

tf.compat.v1.set_random_seed(seed_value)


class data_generator:
    def create_train(dataset_info, batch_size, shape, augument=False):
        assert shape[2] == 3
        while True:
            dataset_info = shuffle(dataset_info, random_state=0)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(
                        X_train_batch[i]['path'], shape)
                    if augument:
                        image = data_generator.augment(image)
                    batch_images.append(image/255.)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    def load_image(path, shape):
        image_red_ch = Image.open(path+'_red.png')
        image_yellow_ch = Image.open(path+'_yellow.png')
        image_green_ch = Image.open(path+'_green.png')
        image_blue_ch = Image.open(path+'_blue.png')
        image = np.stack((
            np.array(image_red_ch),
            np.array(image_green_ch),
            np.array(image_blue_ch)), -1)
        image = cv2.resize(image, (shape[0], shape[1]))
        return image

    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug


def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = InceptionV3(include_top=False,
                             # weights='imagenet',
                             weights=None,
                             input_shape=input_shape)
    # load_model = tf.keras.models.load_model('save_model.h5',compile=False)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    # x = load_model(bn)
    x = Conv2D(32, kernel_size=(1, 1), activation='relu', kernel_initializer=args.kernel_initializer)(x)
    x = Flatten()(x)

    x = Dropout(params['DROPOUT'])(x)
    x = Dense(params['DENSE_UNIT'], activation='relu', kernel_initializer=args.kernel_initializer)(x)
    x = Dropout(params['DROPOUT'])(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model


def main():
    print(f"[INFO] Trial Config: {params}")
    start = time.time()

    # if params["tf_gpu_thread_mode"] in ["global", "gpu_private", "gpu_shared"]:
    # 	os.environ['TF_GPU_THREAD_MODE'] = params["tf_gpu_thread_mode"]

    train_dataset_info = []
    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        train_dataset_info.append({
            'path': os.path.join(path_to_train, name),
            'labels': np.array([int(label) for label in labels])})
    train_dataset_info = np.array(train_dataset_info)

    # split data into train, valid
    indexes = np.arange(train_dataset_info.shape[0])
    # np.random.shuffle(indexes)
    train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=0)
    # create train and valid datagens
    is_augument = args.is_augument == 1
    batch_size = params['BATCH_SIZE'] * NUM_GPU
    train_generator = data_generator.create_train(
        train_dataset_info[train_indexes], batch_size, (SIZE, SIZE, 3), augument=is_augument)
    validation_generator = data_generator.create_train(
        train_dataset_info[valid_indexes], 32, (SIZE, SIZE, 3), augument=False)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        is_load_model = args.is_load_model
        if is_load_model:
            model = tf.keras.models.load_model("../../save/inception_human_save_tpe")
        else:
            model = create_model(input_shape=(SIZE, SIZE, 3), n_out=28)
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
        train_generator,
        steps_per_epoch=np.ceil(float(len(train_indexes)) / batch_size),
        # steps_per_epoch=10,
        validation_data=validation_generator,
        validation_steps=np.ceil(float(len(valid_indexes)) / batch_size),
        # validation_steps=10,
        epochs=epoch,
        # epochs=5,
        verbose=1)
    # keras_model_path = "../../save/inception_human_save_tpe"
    # model.save(keras_model_path)
    end = time.time()
    spent_time = (start - end) / 3600.0
    train_acc = 0. if len(his.history['accuracy']) < 1 else his.history['accuracy'][-1]
    train_top5_acc = 0. if len(his.history['top_k_categorical_accuracy']) < 1 else his.history['top_k_categorical_accuracy'][-1]
    val_acc = 0. if len(his.history['val_accuracy']) < 1 else his.history['val_accuracy'][-1]
    val_top5_acc = 0. if len(his.history['val_top_k_categorical_accuracy']) < 1 else his.history['val_top_k_categorical_accuracy'][-1]
    with open(args.log_path,"a") as f:
        print(train_acc, train_top5_acc, val_acc, val_top5_acc, abs(spent_time)*60, start, end, params, file=f)
    if args.is_soo:
        nni.report_final_result(val_acc)
    else:
        report_dict = {'accuracy': val_acc, 'runtime': spent_time, 'default': val_acc}
        nni.report_final_result(report_dict)


def get_default_params():
    return {
        'BATCH_SIZE': 80,
        'LEARNING_RATE': 0.0001,
        'NUM_EPOCH': 1,
        "DENSE_UNIT": 256,
        'DROPOUT': 0.3,
        'OPTIMIZER': 'adam'
    }


if __name__ == '__main__':
    params = get_default_params()
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)
    # Load dataset info
    path_to_train = '/research/dept7/ychen/data/human/train/'
    data = pd.read_csv('/research/dept7/ychen/data/human/train.csv')
    SIZE = 299
    NUM_GPU = 2
    epoch = params['TRIAL_BUDGET'] if 'TRIAL_BUDGET' in params.keys() else params['NUM_EPOCH']
    main()
