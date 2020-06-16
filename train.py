import logging
import os

import tensorflow as tf
from opts import configure_args
from model.network_architecture import create_model
from utils import set_logger, data_generation, get_partition_label


if __name__ == "__main__":
    args = configure_args()
    set_logger('output/train_{}.log'.format(args.name))
    tf.random.set_seed(100)
    partition, label = get_partition_label()

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

    logging.info('Initializing model...')
    logging.info('Batch size: {}'.format(args.batch_size))

    with strategy.scope():
        if args.restore:
            logging.info('Restoring model...')
            model = tf.keras.models.load_model('convlstm_{}.h5'.format(args.name))
        else:
            model = create_model(args)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(),
                      metrics=['accuracy', tf.keras.metrics.Precision()])

    checkpoint_path = 'training_{}/cp-{epoch:04d}.ckpt'.format(args.name)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        period=5)

    csv_logger = tf.keras.callbacks.CSVLogger('output/model_{}.csv'.format(args.name))
    logging.info('Training model...')

    model.fit(train_dataset, epochs=args.epoch, validation_data=validation_dataset, callbacks=[csv_logger, cp_callback])
    model.save('convlstm_{}.h5'.format(args.name), format='h5')
