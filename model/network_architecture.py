import tensorflow as tf
from tensorflow.keras.layers import Flatten, ConvLSTM2D, BatchNormalization, Dense
from tensorflow.keras import Input, Model


def create_model(args):
    input_image = Input(name='img', shape=(args.frame_size, args.image_size, args.image_size, 3))

    input_mask = Input(name='mask', shape=(args.frame_size,))

    conv_1 = ConvLSTM2D(filters=20, kernel_size=(3, 3), padding='same', return_sequences=True)(input_image,
                                                                                               mask=input_mask)

    batch_1 = BatchNormalization()(conv_1)

    conv_2 = ConvLSTM2D(filters=10, kernel_size=(3, 3), padding='same')(batch_1, mask=input_mask)

    batch_2 = BatchNormalization()(conv_2)

    flat = Flatten()(batch_2)

    dense_1 = Dense(32, activation='relu')(flat)

    output = Dense(2, activation='sigmoid')(dense_1)

    model = Model([input_image, input_mask], output)

    return model
