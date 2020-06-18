import pickle
from itertools import islice

import boto3
from PIL import Image, ExifTags
import logging
import numpy as np
import cv2
import json
from mtcnn.mtcnn import MTCNN
from numpy import asarray


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


def get_partition_label():
    with open('data/label.json') as f:
        label = json.load(f)

    with open('data/partition.json') as f:
        partition = json.load(f)
    return partition, label


def data_generation(user_id_list_temp, label, args):
    s3 = boto3.client('s3')
    # Generates data containing batch_size samples
    data_len = len(user_id_list_temp)
    x = np.empty((data_len, args.frame_size, args.image_size, args.image_size, 3))
    y = np.empty((data_len, 2), dtype=int)
    resized_img = np.empty((args.frame_size, args.image_size, args.image_size, 3))
    mask = np.empty((data_len, args.frame_size), dtype=int)

    # Generate data
    for i, user_id in enumerate(user_id_list_temp):
        # rename error
        y[i] = label[str(user_id)]
        user_id = str(user_id)
        if user_id[:3] == '999' and not (user_id == '9993' or user_id == '9996'):
            user_id = '#' + user_id[3:]

        response = s3.get_object(Bucket='cureskin-dataset', Key='new_data/image_{}.pkl'.format(user_id))
        body = response['Body'].read()
        img_frame = pickle.loads(body)

        for j, img in enumerate(img_frame):
            resized_img[j, ] = cv2.resize(img, dsize=(args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
        del img_frame
        resized_img /= 255
        # (batch, frame, size, size, channel)
        x[i, ] = resized_img
        img_mask = np.all((resized_img == 0), axis=1)
        img_mask = np.all((img_mask == True), axis=1)
        img_mask = np.all((img_mask == True), axis=1)
        img_mask = np.logical_not(img_mask)
        mask[i, ] = img_mask

    for x, mask, y in zip(x, mask, y):
        yield {'img': x, 'mask': mask}, y


def pre_process(img_frame, args):
    x = np.empty((1, args.frame_size, args.image_size, args.image_size, 3))
    mask = np.empty((1, args.frame_size), dtype=int)

    x[0, ] = img_frame
    img_mask = np.all((img_frame == 0), axis=1)
    img_mask = np.all((img_mask == True), axis=1)
    img_mask = np.all((img_mask == True), axis=1)
    img_mask = np.logical_not(img_mask)
    mask[0, ] = img_mask

    return x, mask


def pre_process_post(img_frame, args):
    face_img_frame = np.empty((args.frame_size, args.image_size, args.image_size, 3))
    x = np.empty((1, args.frame_size, args.image_size, args.image_size, 3))
    mask = np.empty((1, args.frame_size), dtype=int)
    detector = MTCNN()

    for i, img in islice(enumerate(img_frame), 0, args.frame_size):
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = img.convert('RGB')
        img = asarray(img)
        results = detector.detect_faces(img)

        if len(results) == 0:
            face_img_frame[i, ] = np.zeros((args.image_size, args.image_size, 3))
        else:
            x1, y1, width, height = results[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = img[y1:y2, x1:x2]
            face = Image.fromarray(face)
            face = face.resize((args.image_size, args.image_size))
            print(face.shape)
            face_img_frame[i, ] = face
    face_img_frame /= 255

    x[0, ] = face_img_frame
    img_mask = np.all((face_img_frame == 0), axis=1)
    img_mask = np.all((img_mask == True), axis=1)
    img_mask = np.all((img_mask == True), axis=1)
    img_mask = np.logical_not(img_mask)
    mask[0, ] = img_mask

    return x, mask


def detect_face(img, args):
    detector = MTCNN()
    img = Image.fromarray(img.astype(np.uint8))
    img = img.convert('RGB')
    img = asarray(img)
    results = detector.detect_faces(img)
    if len(results) == 0:
        return np.zeros((args.image_size, args.image_size, 3))
    else:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]
        face = Image.fromarray(face)
        face = face.resize((args.image_size, args.image_size))
        return face
