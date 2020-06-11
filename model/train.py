import logging
import tensorflow as tf
import json
from opts import configure_args
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, Multiply, Masking
from tensorflow.keras import Input, Model
from tensorflow.keras.applications.resnet50 import ResNet50
from utils import set_logger, DataGenerator

if __name__ == "__main__":
    args = configure_args()
    set_logger('../output/train.log')

    with open('../data/label.json') as f:
        label = json.load(f)

    with open('../data/partition.json') as f:
        partition = json.load(f)

    input_shape = (args.frame_size, args.image_size, args.image_size, 3)

    # Generators
    training_generator = DataGenerator(partition['train'], label, batch_size=args.batch_size, input_shape=input_shape)
    validation_generator = DataGenerator(partition['validation'], label, batch_size=args.batch_size)
    testing_generator = DataGenerator(partition['test'], label, batch_size=args.batch_size)

    logging.info('Initializing model...')


    # model = tf.keras.Sequential()
    # model.add()
    #
    # model.add(tf.keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3),
    #                                      padding='same', return_sequences=True))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    #
    # # model.add(tf.keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3),
    # #                                      padding='same', return_sequences=True))
    # # model.add(tf.keras.layers.BatchNormalization())
    # # #model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    # #
    # # model.add(tf.keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3),
    # #                                      padding='same', return_sequences=True))
    # # model.add(tf.keras.layers.BatchNormalization())
    # # #model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    # #
    # # model.add(tf.keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3),
    # #                                      padding='same', return_sequences=True))
    # # model.add(tf.keras.layers.BatchNormalization())
    # # #model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    # #
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(64))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(2))
    # model.add(tf.keras.layers.Activation('sigmoid'))



    resnet = ResNet50(include_top=False, weights='imagenet', pooling='avg')

    input_layer = Input(shape=input_shape)

    input_mask = Input(shape=(args.frame_size, 1))

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

    history = model.fit(training_generator,
                        validation_data=validation_generator,
                        callbacks=[csv_logger],
                        use_multiprocessing=True,
                        workers=6)
