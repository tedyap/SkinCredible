import os
import logging
from opts import configure_args
from utils import set_logger, data_generation, get_partition_label
import tensorflow as tf
from model.network_architecture import create_model

if __name__ == "__main__":
    args = configure_args()
    set_logger('output/train_{}.log'.format(args.name))
    partition, label = get_partition_label()

    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync
    logging.info('InSync: {}'.format(strategy.num_replicas_in_sync))

    checkpoint_filepath = os.path.join(args.model_dir, 'output/checkpoint')
    latest = tf.train.latest_checkpoint(checkpoint_filepath)

    types = ({'img': tf.float32, 'mask': tf.float32}, tf.float32)
    shapes = (({'img': [args.frame_size, args.image_size, args.image_size, 3], 'mask': [args.frame_size]}), [2])
    test_dataset = tf.data.Dataset.from_generator(
        lambda: data_generation(partition['test'][:args.data_size], label, args),
        output_types=types, output_shapes=shapes).batch(BATCH_SIZE)

    # with strategy.scope():
    #     model = tf.keras.models.load_model('conv_lstm.h5')
    #     model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(),
    #                   metrics=['accuracy', tf.keras.metrics.Precision()])

    checkpoint_path = 'training_{}'.format(args.name) + '/cp-0005.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model = create_model(args)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(),
                                     metrics=['accuracy', tf.keras.metrics.Precision()])
    model.load_weights(checkpoint_path)

    result = model.predict(test_dataset)
    print(result)


