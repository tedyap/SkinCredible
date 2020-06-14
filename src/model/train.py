import logging
import os
import cv2
import tensorflow as tf
import json
from opts import configure_args
from tensorflow.keras.layers import Flatten, ConvLSTM2D, BatchNormalization, MaxPool3D, Dense
from tensorflow.keras import Input, Model
from utils import set_logger, DataGenerator
import boto3
import numpy as np
import pickle
from datetime import datetime


def data_generation(user_id_list_temp, label, args):
    s3 = boto3.client('s3')
    # Generates data containing batch_size samples
    data_len = len(user_id_list_temp)
    x = np.empty((data_len, args.frame_size, args.image_size, args.image_size, 3))
    y = np.empty((data_len, 2), dtype=int)
    mask = np.empty((data_len, args.frame_size), dtype=int)

    # Generate data
    for i, user_id in enumerate(user_id_list_temp):
        response = s3.get_object(Bucket='cureskin-dataset', Key='new_data/image_{}.pkl'.format(user_id))
        body = response['Body'].read()
        img_frame = pickle.loads(body)

        for j, img in enumerate(img_frame):
            img_frame[j, ] = cv2.resize(img, dsize=(args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)

        img_frame /= 255
        # (batch, frame, size, size, channel)
        x[i, ] = img_frame

        img_mask = np.all((img_frame == 0), axis=1)
        img_mask = np.all((img_mask == True), axis=1)
        img_mask = np.all((img_mask == True), axis=1)
        img_mask = np.logical_not(img_mask)
        mask[i, ] = img_mask
        y[i] = label[str(user_id)]

    for x, mask, y in zip(x, mask, y):
        yield {'img': x, 'mask': mask}, y


if __name__ == "__main__":
    args = configure_args()

    set_logger(os.path.join(args.model_dir, 'output/train.log'))

    with open(os.path.join(args.model_dir, 'data/label.json')) as f:
        label = json.load(f)

    with open(os.path.join(args.model_dir, 'data/partition.json')) as f:
        partition = json.load(f)

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            logging.error(e)

    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync
    logging.info('InSync: {}'.format(strategy.num_replicas_in_sync))
    # BATCH_SIZE = args.batch_size

    types = ({'img': tf.float32, 'mask': tf.float32}, tf.float32)

    shapes = (({'img': [args.frame_size, args.image_size, args.image_size, 3], 'mask': [args.frame_size]}), [2])

    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generation(partition['train'][:args.data_size], label, args),
        output_types=types, output_shapes=shapes).batch(BATCH_SIZE)
    validation_dataset = tf.data.Dataset.from_generator(
        lambda: data_generation(partition['validation'][:args.data_size], label, args),
        output_types=types, output_shapes=shapes).batch(BATCH_SIZE)
    #test_dataset = tf.data.Dataset.from_generator(
    #    lambda: data_generation(partition['test'][:args.data_size], label, args),
    #    output_types=types, output_shapes=shapes).batch(BATCH_SIZE)

    logging.info('Initializing model...')
    logging.info('Batch size: {}'.format(args.batch_size))

    logdir = os.path.join(args.model_dir, 'logs/scalars/' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    with strategy.scope():
        input_image = Input(name='img', shape=(args.frame_size, args.image_size, args.image_size, 3))

        input_mask = Input(name='mask', shape=(args.frame_size,))

        conv_1 = ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', return_sequences=True)(input_image,
                                                                                                  mask=input_mask)

        batch_1 = BatchNormalization()(conv_1)

        max_1 = MaxPool3D(pool_size=(1, 2, 2), padding='same')(batch_1)

        flat = Flatten()(max_1)

        dense_1 = Dense(64, activation='relu')(flat)

        output = Dense(2, activation='sigmoid')(dense_1)

        model = Model([input_image, input_mask], output)

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(),
                      metrics=['accuracy', tf.keras.metrics.Precision()])

    checkpoint_filepath = os.path.join(args.model_dir, 'output/checkpoint')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True)

    logging.info('Training model...')
    history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset, callbacks=[tensorboard_callback, model_checkpoint_callback])

    model.save(os.path.join(args.model_dir, 'output/convlstm.h5'))
