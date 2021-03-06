import os
import logging
from model.opts import configure_args
from model.utils import set_logger, data_generation, get_partition_label
import tensorflow as tf

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

    with strategy.scope():
        model = tf.keras.models.load_model('conv_lstm_{}.h5'.format(args.name))
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(),
                      metrics=['accuracy', tf.keras.metrics.Precision()])

    loss, acc, precision = model.evaluate(test_dataset)
    logging.info("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    logging.info("Restored model, precision: {:5.2f}%".format(100 * precision))
