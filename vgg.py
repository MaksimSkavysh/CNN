from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader


print('\nVGG network with 128x128 images input\n')

# Data loading and preprocessing
ITERATION = 3

FOLDER_TO_SAVE = './out/vgg_im128_i' + str(ITERATION)
# FOLDER_TO_LOAD = './out/vgg_im128_i2/checkpoints-74020'
FOLDER_TO_LOAD = './out/vgg_im128_i3/checkpoints-111030'

TRAIN_DATA = './train_data'
VAL_DATA = './val_data'

X, Y = image_preloader(TRAIN_DATA,
                       image_shape=(128, 128),
                       mode='file',
                       files_extension=['.png'])

X_val, Y_val = image_preloader(
    VAL_DATA,
    image_shape=(128, 128),
    mode='file',
    files_extension=['.png'],
)

# Building 'VGG Network'
network = input_data(shape=[None, 128, 128, 3])

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

# Training
model = tflearn.DNN(network,
                    checkpoint_path=FOLDER_TO_SAVE + '/checkpoints/',
                    tensorboard_dir=FOLDER_TO_SAVE + '/logs/',
                    best_checkpoint_path=FOLDER_TO_SAVE + '/best_checkpoint',
                    best_val_accuracy=0.8,
                    max_checkpoints=2,
                    tensorboard_verbose=0)


# # print('\nStart loading ' + FOLDER_TO_LOAD + ' ... ')
# # model.load(FOLDER_TO_LOAD)

# print('\nStart loading ' + FOLDER_TO_LOAD + ' ... ')
# model.load(FOLDER_TO_LOAD)

print('\nStart training ...')
model.fit(X, Y,
          n_epoch=20,
          validation_set=(X_val, Y_val),
          shuffle=True,
          show_metric=True,
          batch_size=32,
          snapshot_step=500,
          snapshot_epoch=True,
          run_id='vgg_TB5')

