from __future__ import division, print_function, absolute_import
from tflearn.data_utils import image_preloader
from sklearn import metrics
import matplotlib.pyplot as plt
from models.alex import get_alex_model


ITERATION = 2
TRAIN_DATA = './train_data'
VAL_DATA = './val_data'


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
    model = get_alex_model(
        filter_size=filter_size,
        folder_to_save='not_meter',
        folder_to_load=folder_to_load,
        image_size=image_size,
        strides=strides,
        learning_rate=0.0003,
    )

    arr_pred, arr_val = predict_all(model, x_val, y_val)
    calculate_and_print_roc(arr_val, arr_pred, title=title)


alex_size_params = [
    ("AlexNet 32x32", "./out/alex_s32_2/model/model", 32),
    ("AlexNet 64x64", "./out/alex_s64_2/model/model", 64),
    ("AlexNet 128x128", "./out/alex_s128_2/model/model", 128),
    ("AlexNet 160x160", "./out/alex_s160_2/model/model", 160),
    ("AlexNet 256x256", "./out/alex_s256_2/checkpoints-36114", 256),
]


def build_auc():
    i = 4
    alexnet_rocauc(
        title=alex_size_params[i][0],
        image_size=alex_size_params[i][2],
        folder_to_load=alex_size_params[i][1],
    )

    plt.show()


build_auc()
