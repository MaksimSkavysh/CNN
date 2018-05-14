dataset_file = 'my_dataset.txt'

# Build a HDF5 dataset (only required once)
from tflearn.data_utils import build_hdf5_image_dataset


TRAIN_DATA = '../train_data'
VAL_DATA = '../val_data'


build_hdf5_image_dataset(TRAIN_DATA,
                         image_shape=(128, 128),
                         mode='file',
                         output_path='dataset_train.h5',
                         categorical_labels=True,
                         normalize=True
                         )
