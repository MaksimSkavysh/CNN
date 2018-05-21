from __future__ import division, print_function, absolute_import
import sys

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt

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


def filter_size_run_cnn(filter_size):
    image_size = 128
    strides = 4
    if filter_size < 9:
        strides = 2

    print('\n\n\nstart with filter size: ' + str(filter_size) + '; and strides: ' + str(strides))
    folder_to_save = './out/alex_f' + str(filter_size) + '_' + str(ITERATION)
    folder_to_load = './out/alex_s128_1/model/alex_model'
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
    x_val = x_val[:500]
    y_val = y_val[:500]
    y_pred = model.predict_label(x_val)

    print('\nCalculating AUC ROC ...')

    # arr_pred = [y[0] for y in y_pred]
    # arr_val = [y[1] for y in y_val]
    # res = roc_auc_score(arr_val, arr_pred)
    # # res = tflearn.objectives.roc_auc_score(y_pred, y_pred)
    # print(res)

    fpr, tpr, threshold = metrics.roc_curve(y_val, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


filter_size_run_cnn(11)
