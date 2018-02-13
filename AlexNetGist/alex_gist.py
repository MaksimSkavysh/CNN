from __future__ import division, print_function, absolute_import


# import numpy as np
# from numpy import array
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader

# dataset = './data'
dataset_tr = '/media/maksim/TomD/datasets/Histology_CAMELYON16_300K_Tiles/train1'
dataset_pr = '/media/maksim/TomD/datasets/Histology_CAMELYON16_300K_Tiles/valid1'
# datafile = './data/index.txt'
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
network = conv_2d(network, 96, 11, strides=4, activation='relu')
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
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='momentum', loss='categorical_crossentropy',
                     learning_rate=0.002)

# Training
model = tflearn.DNN(network,
                    checkpoint_path='../AlexGistInfo/checkpoints/',
                    tensorboard_dir='../AlexGistInfo/logs/',
                    best_checkpoint_path='../AlexGistInfo/best_model_alexnet',
                    max_checkpoints=3,
                    tensorboard_verbose=0)


print('Start training ...')
model.fit(X, Y, n_epoch=5,
          validation_set=(X_pr, Y_pr),
          shuffle=True, batch_size=64,
          show_metric=True,
          snapshot_epoch=True,
          run_id='alexnet_gist')

model.save('../AlexGistInfo/alex_gist.tflearn')
