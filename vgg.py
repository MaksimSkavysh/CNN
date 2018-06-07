from __future__ import division, print_function, absolute_import
from tflearn.data_utils import image_preloader
from models.vgg import get_vgg_model
import numpy as np


print('\nVGG network with 128x128 images input\n')

# Data loading and preprocessing
ITERATION = 3

# TRAIN_DATA = './train_data'
# VAL_DATA = './val_data'
TRAIN_DATA = './train_data_grey'
VAL_DATA = './val_data_grey'


# FOLDER_TO_SAVE = './out/vgg_im128_i' + str(ITERATION)
# FOLDER_TO_LOAD = './out/vgg_im128_i2/checkpoints-74020'
# FOLDER_TO_LOAD = './out/vgg_im128_i3/checkpoints-111030'
FOLDER_TO_SAVE = './out_grey/vgg_i1'
FOLDER_TO_LOAD = ''


X, Y = image_preloader(TRAIN_DATA,
                       grayscale=True,
                       image_shape=(128, 128),
                       mode='file',
                       files_extension=['.png'])

X_val, Y_val = image_preloader(
    VAL_DATA,
    grayscale=True,
    image_shape=(128, 128),
    mode='file',
    files_extension=['.png'],
)

X = np.reshape(X, (len(X), 128, 128, 1))
X_val = np.reshape(X_val, (len(X_val), 128, 128, 1))

model = get_vgg_model(foler_to_save=FOLDER_TO_SAVE, folder_to_load='', depth=1)
print('\nStart training ...')
model.fit(X, Y,
          n_epoch=50,
          validation_set=(X_val, Y_val),
          shuffle=True,
          show_metric=True,
          batch_size=32,
          snapshot_step=500,
          snapshot_epoch=True,
          run_id='vgg_TB5')

