from __future__ import division, print_function, absolute_import
import numpy as np

from tflearn.data_utils import image_preloader
from models.zf import get_zf_model

ITERATION = 2
# TRAIN_DATA = './train_data'
# VAL_DATA = './val_data'
TRAIN_DATA = './train_data_grey'
VAL_DATA = './val_data_grey'


def filter_size_run_cnn():
    image_size = 128
    strides = 2
    batch_size = 16

    print('\n\n\nstart with filter size: ')
    # folder_to_save = './out/zfnet_' + str(ITERATION)
    # folder_to_load = './out/zfnet_1/checkpoints-96226'
    folder_to_save = './out_grey/zfnet_' + str(ITERATION)
    # folder_to_load = './out_grey/zfnet_' + str(ITERATION-1) + '/model/model'
    folder_to_load = ''

    x, y = image_preloader(TRAIN_DATA,
                           grayscale=True,
                           image_shape=(image_size, image_size),
                           mode='file',
                           files_extension=['.png'])
    x_val, y_val = image_preloader(VAL_DATA,
                                   grayscale=True,
                                   image_shape=(image_size, image_size),
                                   mode='file',
                                   files_extension=['.png'])
    x = np.reshape(x, (len(x), 128, 128, 1))
    x_val = np.reshape(x_val, (len(x_val), 128, 128, 1))

    model = get_zf_model(
        filter_size=7,
        folder_to_save=folder_to_save,
        folder_to_load=folder_to_load,
        image_size=image_size,
        strides=strides,
        learning_rate=0.0003,
        channels=1,
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


filter_size_run_cnn()
