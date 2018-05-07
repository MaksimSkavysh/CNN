from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader

# Data loading and preprocessing
ITERATION = 1

FOLDER_TO_SAVE = './out/vgg_i' + str(ITERATION)
# FOLDER_TO_LOAD = './out/vgg_i' + str(ITERATION-1) + '/model/alex_model'

TRAIN_DATA = './train_data'
VAL_DATA = './val_data'

X, Y = image_preloader(TRAIN_DATA,
                       image_shape=(224, 224),
                       mode='file',
                       normalize=True,
                       files_extension=['.png'])

# X_val, Y_val = image_preloader(
#     VAL_DATA,
#     image_shape=(224, 224),
#     mode='file',
#     files_extension=['.png'],
# )


# import tflearn.datasets.oxflower17 as oxflower17
# X, Y = oxflower17.load_data(one_hot=True)

# Building 'VGG Network'
network = input_data(shape=[None, 224, 224, 3])

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
network = fully_connected(network, 17, activation='softmax')

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
                    tensorboard_verbose=1)


# # print('\nStart loading ' + FOLDER_TO_LOAD + ' ... ')
# # model.load(FOLDER_TO_LOAD)

print('\nStart training ...')
model.fit(X, Y,
          n_epoch=1,
          shuffle=True,
          show_metric=True,
          batch_size=8,
          snapshot_step=500,
          snapshot_epoch=True,
          run_id='vgg_TB5')

