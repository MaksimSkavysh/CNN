from __future__ import division, print_function, absolute_import
from tflearn.data_utils import image_preloader
from models.vgg import get_vgg_model


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


model = get_vgg_model(foler_to_save=FOLDER_TO_SAVE, folder_to_load='')
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

