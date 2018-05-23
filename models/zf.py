import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def get_zf_model(
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
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 1024, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
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
