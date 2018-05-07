from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import image_preloader

import alex_model
import visualization

# FOLDER_TO_LOAD = '../out/alex_13_mac/checkpoints-24562'
FOLDER_TO_LOAD = '../out/alex_13_mac/model/alex_model'
FOLDER_TO_SAVE = '../out/alex_13_mac'

dataset_tr = '../small_data/valid'
X, Y = image_preloader(dataset_tr,
                       image_shape=(227, 227),
                       mode='folder',
                       files_extension=['.png'])

network, conv_2d_1 = alex_model.get_network()

# Training
model = tflearn.DNN(network,
                    checkpoint_path=FOLDER_TO_SAVE + '/checkpoints/',
                    tensorboard_dir=FOLDER_TO_SAVE + '/logs/',
                    best_checkpoint_path=FOLDER_TO_SAVE + '/best_checkpoint',
                    max_checkpoints=2,
                    best_val_accuracy=0.5,
                    tensorboard_verbose=0)


print('Start loading ' + FOLDER_TO_LOAD)
model.load(FOLDER_TO_LOAD)

# print('Start evaluating')
# res1 = model.evaluate(X, Y)
# print('Train examlpes: ', res1)

print('start visualization')
visualization.display_convolutions(model, conv_2d_1)
