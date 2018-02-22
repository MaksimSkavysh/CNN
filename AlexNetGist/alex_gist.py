from __future__ import division, print_function, absolute_import
# import numpy as np
# from numpy import array
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader
from tflearn.initializations import variance_scaling


FOLDER_TO_LOAD = '../out/alex_3_mac/model/alex_model'
FOLDER_TO_SAVE = '../out/alex_4_mac'

# dataset = './data'
# dataset_tr = '/media/maksim/TomD/datasets/Histology_CAMELYON16_300K_Tiles/train1'
# dataset_pr = '/media/maksim/TomD/datasets/Histology_CAMELYON16_300K_Tiles/valid1'
dataset_tr = '../small_data/train'
dataset_pr = '../small_data/valid'

X, Y = image_preloader(dataset_tr,
                       image_shape=(227, 227),
                       mode='folder',
                       files_extension=['.png'])

X_pr, Y_pr = image_preloader(dataset_pr,
                             image_shape=(227, 227),
                             mode='folder',
                             files_extension=['.png'])


print('Start building ...')
# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 4])
# network = conv_2d(network, 96, 11, strides=4, activation='relu')

network = conv_2d(network,
                  nb_filter=96,
                  filter_size=11,
                  strides=4,
                  activation='relu',
                  regularizer='L2',
                  weight_decay=0.0005,
                  bias_init='uniform',
                  trainable=True,
                  restore=True)
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')

network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network,
                     optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005)

# Training
model = tflearn.DNN(network,
                    checkpoint_path=FOLDER_TO_SAVE + '/checkpoints/',
                    tensorboard_dir=FOLDER_TO_SAVE + '/logs/',
                    best_checkpoint_path=FOLDER_TO_SAVE + '/best_checkpoint',
                    max_checkpoints=2,
                    best_val_accuracy=0.5,
                    tensorboard_verbose=0)


print('Start loading ' + FOLDER_TO_LOAD + ' ... ')
model.load(FOLDER_TO_LOAD)

print('Start training ...')
model.fit(X,
          Y,
          n_epoch=3,
          validation_set=(X_pr, Y_pr),
          shuffle=True,
          batch_size=128,
          show_metric=True,
          snapshot_epoch=True,
          run_id='alexnet_gist')

model.save(FOLDER_TO_SAVE + '/model/alex_model')
