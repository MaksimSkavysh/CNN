from __future__ import division, print_function, absolute_import
import numpy as np

from tflearn.data_utils import image_preloader
from models.alex import get_alex_model

ITERATION = 1
TRAIN_DATA = './train_data_grey'
VAL_DATA = './val_data_grey'
# TRAIN_DATA = './train_data'
# VAL_DATA = './val_data'


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
    folder_to_save = './out_grey/alex_f' + str(filter_size) + '_' + str(ITERATION)
    # folder_to_load = './out/alex_f' + str(filter_size) + '_' + str(ITERATION-1) + '/model/model'
    folder_to_load = ''

    x, y = image_preloader(TRAIN_DATA,
                           image_shape=(image_size, image_size),
                           grayscale=True,
                           # filter_channel=True,
                           mode='file',
                           files_extension=['.png'])
    x_val, y_val = image_preloader(VAL_DATA,
                                   image_shape=(image_size, image_size),
                                   grayscale=True,
                                   # filter_channel=True,
                                   mode='file',
                                   files_extension=['.png'])
    print('loaded_data')
    x = np.reshape(x, (len(x), 128, 128, 1))
    x_val = np.reshape(x_val, (len(x_val), 128, 128, 1))

    model = get_alex_model(
        filter_size=filter_size,
        folder_to_save=folder_to_save,
        folder_to_load=folder_to_load,
        image_size=image_size,
        strides=strides,
        learning_rate=0.0003,
        depth=1,
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


filter_size_run_cnn(11)

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

