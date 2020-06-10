import pickle
import boto3
from PIL import Image, ExifTags
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def rotate_image(img):
    try:
        image = Image.open(img)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        return image
    except (AttributeError, KeyError, IndexError):
        image = Image.open(img)
        return image


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, user_id_list, labels, batch_size=32, dim=(30, 160, 160), n_channels=3,
                 n_classes=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.user_id_list = user_id_list
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # denotes the number of batches per epoch
        return int(np.floor(len(self.user_id_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        user_id_list_temp = [self.user_id_list[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(user_id_list_temp)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.user_id_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, user_id_list_temp):
        s3 = boto3.client('s3')
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, user_id in enumerate(user_id_list_temp):
            response = s3.get_object(Bucket='cureskin-dataset', Key='data/image_{}.pkl'.format(user_id))
            body = response['Body'].read()
            img = pickle.loads(body)
            img /= 255
            x[i, ] = img
            y[i] = self.labels[str(user_id)]

        return x, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
