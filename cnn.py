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

ITERATION = 5

FOLDER_TO_SAVE = './out/iteration' + str(ITERATION)
FOLDER_TO_LOAD = './out/iteration' + str(ITERATION-1) + '/model/alex_model'

TRAIN_DATA = './train_data'
VAL_DATA = './val_data'

X, Y = image_preloader(TRAIN_DATA,
                       image_shape=(128, 128),
                       mode='file',
                       files_extension=['.png'])

X_val, Y_val = image_preloader(VAL_DATA,
                       image_shape=(128, 128),
                       mode='file',
                       files_extension=['.png'])


print('Start building ...')
network = input_data(shape=[None, 128, 128, 3])
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
                    max_checkpoints=3,
                    best_val_accuracy=0.6,
                    tensorboard_verbose=0)


print('\nStart loading ' + FOLDER_TO_LOAD + ' ... ')
model.load(FOLDER_TO_LOAD)

print('\nStart training ...')
model.fit(X,
          Y,
          n_epoch=100,
          validation_set=(X_val, Y_val),
          shuffle=True,
          batch_size=128,
          show_metric=True,
          snapshot_epoch=True,
          run_id='alex_TB5')

print('\nStart saving ...')
model.save(FOLDER_TO_SAVE + '/model/alex_model')
