from __future__ import division, print_function, absolute_import
from tflearn.data_utils import image_preloader
from sklearn import metrics
import matplotlib.pyplot as plt
from models.alex import get_alex_model
from models.zf import get_zf_model
from models.vgg import get_vgg_model
import numpy as np


ITERATION = 2
# TRAIN_DATA = './train_data'
# VAL_DATA = './val_data'
TRAIN_DATA = './train_data_grey'
VAL_DATA = './val_data_grey'


def calculate_and_print_roc(
        arr_val,
        arr_pred,
        title='',
        color='b',
):
    # res = roc_auc_score(arr_val, arr_pred)
    # # res = tflearn.objectives.roc_auc_score(y_pred, y_pred)
    # print(res)

    fpr, tpr, threshold = metrics.roc_curve(arr_val, arr_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print(title, roc_auc)

    # method I: plt
    plt.title('ROC ' + title)
    plt.plot(fpr, tpr, color=color)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


def predict_all(model, x_val, y_val, step=500):
    labels_num = len(x_val)
    print('\nStart predicting,  total:' + str(labels_num))

    arr_pred = []
    arr_val = []
    # labels_num = 2000
    for i in range(step, labels_num, step):
        print('step: ' + str(i) + '/' + str(labels_num))
        y_pred = model.predict(x_val[i-step:i])
        # y_pred = model.predict_label(x_val[i-step:i])
        arr_pred = arr_pred + [y[0] for y in y_pred]
        arr_val = arr_val + [y[0] for y in y_val[i-step:i]]
    return arr_pred, arr_val


def alexnet_rocauc(
    image_size=128,
    strides=4,
    filter_size=11,
    folder_to_load='./out/alex_s128_2/model/model',
    title='alex net 160 11',
):
    x_val, y_val = image_preloader(VAL_DATA,
                                   image_shape=(image_size, image_size),
                                   mode='file',
                                   files_extension=['.png'])

    print('loaded_data')
    x_val = np.reshape(x_val, (len(x_val), 128, 128, 1))

    model = get_alex_model(
        filter_size=filter_size,
        folder_to_save='not_meter',
        folder_to_load=folder_to_load,
        image_size=image_size,
        strides=strides,
        learning_rate=0.0003,
        depth=1,
    )

    arr_pred, arr_val = predict_all(model, x_val, y_val)
    calculate_and_print_roc(arr_val, arr_pred, title=title)


def zfnet_rocauc(
    image_size=128,
    strides=2,
    folder_to_load='./out_grey/zfnet_2/checkpoints-296040',
    title='ZF Net',
):
    x_val, y_val = image_preloader(VAL_DATA,
                                   image_shape=(image_size, image_size),
                                   mode='file',
                                   files_extension=['.png'])

    print('loaded_data')
    x_val = np.reshape(x_val, (len(x_val), 128, 128, 1))

    model = get_zf_model(
        filter_size=7,
        folder_to_save='not_meter',
        folder_to_load=folder_to_load,
        image_size=image_size,
        strides=strides,
        learning_rate=0.0003,
        channels=1,
    )

    arr_pred, arr_val = predict_all(model, x_val, y_val, step=200)
    calculate_and_print_roc(arr_val, arr_pred, title=title)


def vgg_rocauc(
    folder_to_load='./out_grey/vgg_i1/checkpoints-185050',
    title='VGG Net',
):
    x_val, y_val = image_preloader(VAL_DATA,
                                   image_shape=(128, 128),
                                   mode='file',
                                   files_extension=['.png'])

    print('loaded_data')
    x_val = np.reshape(x_val, (len(x_val), 128, 128, 1))

    model = get_vgg_model(foler_to_save='not_meter', folder_to_load=folder_to_load, depth=1)

    arr_pred, arr_val = predict_all(model, x_val, y_val, step=200)
    calculate_and_print_roc(arr_val, arr_pred, title=title)


alex_size_params = [
    ("AlexNet 32x32", "./out/alex_s32_2/model/model", 32),
    ("AlexNet 64x64", "./out/alex_s64_2/model/model", 64),
    ("AlexNet 128x128", "./out/alex_s128_2/model/model", 128),
    ("AlexNet 160x160", "./out/alex_s160_2/model/model", 160),
    ("AlexNet 256x256", "./out/alex_s256_2/checkpoints-36114", 256),
]


alex_filter_params = [
    ("AlexNet filter size 3", "./out/alex_f3_1/model/model", 3, 2),
    ("AlexNet filter size 5", "./out/alex_f5_1/model/model", 5, 2),
    ("AlexNet filter size 7", "./out/alex_f7_1/model/model", 7, 2),
    ("AlexNet filter size 9", "./out/alex_f9_1/model/model", 9, 4),
    ("AlexNet filter size 11", "./out/alex_f11_1/model/model", 11, 4),
]


def build_auc():
    # i = 4
    # alexnet_rocauc(
    #     title=alex_size_params[i][0],
    #     image_size=alex_size_params[i][2],
    #     folder_to_load=alex_size_params[i][1],
    # )
    # i = 4
    # alexnet_rocauc(
    #     title=alex_filter_params[i][0],
    #     folder_to_load=alex_filter_params[i][1],
    #     filter_size=alex_filter_params[i][2],
    #     strides=alex_filter_params[i][3],
    # )

    # alexnet_rocauc(
    #     title='Gray scale',
    #     image_size=128,
    #     folder_to_load='out_grey/alex_f11_1/model/model',
    # )
    # zfnet_rocauc()
    vgg_rocauc()

    plt.show()


build_auc()
