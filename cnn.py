from __future__ import division, print_function, absolute_import
import sys

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader

ITERATION = 2
TRAIN_DATA = './train_data'
VAL_DATA = './val_data'


def get_alex_model(
        filter_size,
        folder_to_save,
        folder_to_load,
        image_size=128,
        strides=4,
        learning_rate=0.0003,
):
    print('Start building ...')
    network = input_data(shape=[None, image_size, image_size, 3])
    network = conv_2d(network,
                      nb_filter=96,
                      filter_size=filter_size,
                      strides=strides,
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
                         learning_rate=learning_rate)

    model = tflearn.DNN(network,
                        checkpoint_path=folder_to_save + '/checkpoints/',
                        tensorboard_dir=folder_to_save + '/logs/',
                        best_checkpoint_path=folder_to_save + '/best_checkpoint',
                        max_checkpoints=3,
                        best_val_accuracy=0.6,
                        tensorboard_verbose=0)
    if folder_to_load:
        print('\nStart loading ' + folder_to_load + ' ... ')
        model.load(folder_to_load)

    return model


def img_size_run_cnn(size):
    print('\n\n\nstart with size: ' + str(size))
    folder_to_save = './out/alex_s' + str(size) + '_' + str(ITERATION)
    folder_to_load = './out/alex_s' + str(size) + '_' + str(ITERATION-1) + '/model/alex_model'
    x, y = image_preloader(TRAIN_DATA,
                           image_shape=(size, size),
                           mode='file',
                           files_extension=['.png'])
    x_val, y_val = image_preloader(VAL_DATA,
                                   image_shape=(size, size),
                                   mode='file',
                                   files_extension=['.png'])
    model = get_alex_model(
        filter_size=size,
        folder_to_save=folder_to_save,
        folder_to_load=folder_to_load,
        image_size=size,
        strides=4,
        learning_rate=0.0003,
    )
    print('\nStart training ...')
    model.fit(x,
              y,
              n_epoch=40,
              validation_set=(x_val, y_val),
              shuffle=True,
              batch_size=128,
              show_metric=True,
              snapshot_epoch=True,
              run_id='alex_TB5')
    print('\nStart saving ...')
    model.save(folder_to_save + '/model/model')

# python3.6 ./cnn.py 32 &> cnn_32.txt; python3.6 ./cnn.py 64 &> cnn_64.txt; python3.6 ./cnn.py 128 &> cnn_128.txt; python3.6 ./cnn.py 160 &> cnn_160.txt; python3.6 ./cnn.py 256 &> cnn_256.txt


def filter_size_run_cnn(filter_size):
    image_size = 128
    strides = 4
    batch_size = 128
    if filter_size < 9:
        strides = 2

    print('\n\n\nstart with filter size: ' + str(filter_size) + '; and strides: ' + str(strides))
    folder_to_save = './out/alex_f' + str(filter_size) + '_' + str(ITERATION)
    folder_to_load = './out/alex_f' + str(image_size) + '_' + str(ITERATION-1) + '/model/alex_model'

    x, y = image_preloader(TRAIN_DATA,
                           image_shape=(image_size, image_size),
                           mode='file',
                           files_extension=['.png'])
    x_val, y_val = image_preloader(VAL_DATA,
                                   image_shape=(image_size, image_size),
                                   mode='file',
                                   files_extension=['.png'])

    model = get_alex_model(
        filter_size=filter_size,
        folder_to_save=folder_to_save,
        folder_to_load=folder_to_load,
        image_size=image_size,
        strides=strides,
        learning_rate=0.0003,
    )
    print('\nStart training ...')
    model.fit(x,
              y,
              n_epoch=50,
              validation_set=(x_val, y_val),
              shuffle=True,
              batch_size=batch_size,
              show_metric=True,
              snapshot_epoch=True,
              run_id='alex_TB5')
    print('\nStart saving ...')
    model.save(folder_to_save + '/model/model')

# python3.6 ./cnn.py 3 &> cnn_filter_3.txt; python3.6 ./cnn.py 5 &> cnn_filter_5.txt; python3.6 ./cnn.py 7 &> cnn_filter_7.txt; python3.6 ./cnn.py 9 &> cnn_filter_9.txt; python3.6 ./cnn.py 11 &> cnn_filter_11.txt;
# python3.6 ./cnn.py 11 &> cnn_filter_11.txt; python3.6 ./cnn.py 3 &> cnn_filter_3.txt


# def main():
#     print(sys.argv)
#     if len(sys.argv) > 1:
#         size = int(sys.argv[1])
#         print(size)
#         filter_size_run_cnn(size)
#
# main()

filter_size_run_cnn(3)
