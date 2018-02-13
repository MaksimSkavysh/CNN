from __future__ import division, print_function, absolute_import
from os import listdir

def extract_images():
    DATA_FOLDER = './data'
    images = listdir(DATA_FOLDER)

    file = open('./data_index.txt', 'w')
    for img in images:
        class_type = 1 if 'Normal' in img else 0
        file.write('./data/' + img + ' ' + str(class_type) + '\n')
    file.close()

extract_images()
