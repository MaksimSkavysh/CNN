from __future__ import division, print_function, absolute_import
from os import listdir
from shutil import copyfile

DATA_FOLDER = '/media/maksim/TomE/datasets/Histology_CAMELYON16_300K_Tiles/Blue_Tiles/Normal/'
DIST_FOLDER = '/media/maksim/TomE/datasets/Histology_CAMELYON16_300K_Tiles/Blue_Tiles/normalWithoutBKG/'


def separate_images(folder):
    images = listdir(folder)
    nrm_files = [x for x in images if "NRM" in x]
    print(nrm_files)
    for file in nrm_files:
        copyfile(folder + file, DIST_FOLDER + file)


separate_images(DATA_FOLDER)
