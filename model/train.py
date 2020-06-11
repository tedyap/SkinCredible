import logging
import tensorflow as tf
import json
from opts import configure_args
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, Multiply, Masking
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
    mask = np.empty((data_len, args.frame_size, 1), dtype=int)

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
        img_mask = np.expand_dims(img_mask, axis=1)

        mask[i,] = img_mask
        y[i] = label[str(user_id)]

    return {'img': x, 'mask': mask}, tf.keras.utils.to_categorical(y, num_classes=2)


if __name__ == "__main__":
    args = configure_args()
    set_logger('../output/train.log')

    with open('../data/label.json') as f:
        label = json.load(f)

    with open('../data/partition.json') as f:
        partition = json.load(f)

    input_shape = (args.frame_size, args.image_size, args.image_size, 3)

    # Generators
    # training_generator = DataGenerator(partition['train'], label, batch_size=args.batch_size, input_shape=input_shape)
    # validation_generator = DataGenerator(partition['validation'], label, batch_size=args.batch_size)
    # testing_generator = DataGenerator(partition['test'], label, batch_size=args.batch_size)

    #x_train, y_train = data_generation(partition['train'], label, args)
    #x_val, y_val = data_generation(partition['validation'], label, args)
    #x_test, y_test = data_generation(partition['test'], label, args)
    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync
    logging.info('InSync: {}'.format(strategy.num_replicas_in_sync))

    train_dataset = tf.data.Dataset.from_tensor_slices((data_generation(partition['train'][:args.data_size], label, args))).batch(BATCH_SIZE)
    validation_dataset = tf.data.Dataset.from_tensor_slices((data_generation(partition['validation'][:args.data_size], label, args))).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((data_generation(partition['test'][:args.data_size], label, args))).batch(BATCH_SIZE)

    logging.info('Initializing model...')
    logging.info(args.batch_size)

    with strategy.scope():
        resnet = ResNet50(include_top=False, weights='imagenet', pooling='avg')

        input_layer = Input(name='img', shape=input_shape)

        input_mask = Input(name='mask', shape=(args.frame_size, 1))

        curr_layer = TimeDistributed(resnet)(input_layer)

        resnet_output = Dropout(0.5)(curr_layer)

        curr_layer = Multiply()([resnet_output, input_mask])

        cnn_output = curr_layer

        curr_layer = Masking(mask_value=0.0)(curr_layer)

        lstm_out = LSTM(256, dropout=0.5)(curr_layer)
        output = Dense(2, activation='sigmoid')(lstm_out)

        model = Model([input_layer, input_mask], output)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    logging.info('Training model...')
    csv_logger = tf.keras.callbacks.CSVLogger('../output/model.log')

    history = model.fit(train_dataset,
                        validation_data=validation_dataset,
                        callbacks=[csv_logger],
                        use_multiprocessing=True,
                        workers=6)
