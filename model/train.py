import logging
import tensorflow as tf
import json
from opts import configure_args
from tensorflow.keras.layers import Flatten, ConvLSTM2D, BatchNormalization, MaxPool3D, Dense
from tensorflow.keras import Input, Model
from tensorflow.keras.applications.resnet50 import ResNet50
from utils import set_logger, DataGenerator
import boto3
import numpy as np
import pickle

def data_generation(user_id_list_temp, label, args):
    s3 = boto3.client('s3')
    # Generates data containing batch_size samples
    # Initialization
    data_len = len(user_id_list_temp)
    x = np.empty((data_len, args.frame_size, args.image_size, args.image_size, 3))
    y = np.empty((data_len), dtype=int)
    mask = np.empty((data_len, args.frame_size), dtype=int)

    # Generate data
    for i, user_id in enumerate(user_id_list_temp):
        response = s3.get_object(Bucket='cureskin-dataset', Key='data/image_{}.pkl'.format(user_id))
        body = response['Body'].read()
        img = pickle.loads(body)
        img /= 255
        # frame, sz, sz, channel
        x[i,] = img

        img_mask = np.all((img == 0), axis=1)
        img_mask = np.all((img_mask == True), axis=1)
        img_mask = np.all((img_mask == True), axis=1)

        mask[i,] = img_mask
        y[i] = label[str(user_id)]

    return {'img': x, 'mask': mask}, tf.keras.utils.to_categorical(y, num_classes=2)


if __name__ == "__main__":
    args = configure_args()
    #set_logger('../output/train.log')

    with open('data/label.json') as f:
        label = json.load(f)

    with open('data/partition.json') as f:
        partition = json.load(f)

    input_shape = (args.frame_size, args.image_size, args.image_size, 3)

    # Generators
    # training_generator = DataGenerator(partition['train'], label, batch_size=args.batch_size, input_shape=input_shape)
    # validation_generator = DataGenerator(partition['validation'], label, batch_size=args.batch_size)
    # testing_generator = DataGenerator(partition['test'], label, batch_size=args.batch_size)

    #x_train, y_train = data_generation(partition['train'], label, args)
    #x_val, y_val = data_generation(partition['validation'], label, args)
    #x_test, y_test = data_generation(partition['test'], label, args)
    #strategy = tf.distribute.MirroredStrategy()
    #BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync
    #logging.info('InSync: {}'.format(strategy.num_replicas_in_sync))
    BATCH_SIZE = args.batch_size

    train_dataset = tf.data.Dataset.from_tensor_slices((data_generation(partition['train'][:args.data_size], label, args))).batch(BATCH_SIZE)
    validation_dataset = tf.data.Dataset.from_tensor_slices((data_generation(partition['validation'][:args.data_size], label, args))).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((data_generation(partition['test'][:args.data_size], label, args))).batch(BATCH_SIZE)

    logging.info('Initializing model...')
    logging.info(args.batch_size)

    #with strategy.scope():

    input_image = Input(name='img', shape=input_shape)

    input_mask = Input(name='mask', shape=(args.frame_size,))

    conv_1 = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True)(input_image, mask=input_mask)

    batch_1 = BatchNormalization()(conv_1)

    max_1 = MaxPool3D(pool_size=(1, 2, 2), padding='same')(batch_1)

    flat = Flatten()(max_1)

    dense_1 = Dense(64, activation='relu')(flat)

    output = Dense(2, activation='sigmoid')(dense_1)

    model = Model([input_image, input_mask], output)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    logging.info('Training model...')
    csv_logger = tf.keras.callbacks.CSVLogger('output/model.log')

    history = model.fit(train_dataset,
                        validation_data=validation_dataset,
                        callbacks=[csv_logger])
