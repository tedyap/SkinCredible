import boto3
import pickle
import cv2
from model.opts import configure_args
from model.utils import set_logger, pre_process
import numpy as np
from model.network_architecture import create_model
import tensorflow as tf

if __name__ == "__main__":
    user_id = 4642
    args = configure_args()
    set_logger('output/train_{}.log'.format(args.name))
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='cureskin-dataset', Key='new_data/image_{}.pkl'.format(user_id))
    body = response['Body'].read()
    img_frame = pickle.loads(body)

    x, mask = pre_process(img_frame, args)
    print(mask)

    checkpoint_path = 'ckpts'.format(args.name) + '/cp-0005.ckpt'
    model = create_model(args)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(),
                  metrics=['accuracy', tf.keras.metrics.Precision()])
    model.load_weights(checkpoint_path)

    prob = model.predict_on_batch([x, mask])
    prob_class = np.argmax(prob)
