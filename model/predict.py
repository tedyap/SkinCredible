import boto3
import pickle
import cv2
from model.opts import configure_args
from model.utils import set_logger
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
    x = np.empty((1, args.frame_size, args.image_size, args.image_size, 3))
    mask = np.empty((1, args.frame_size), dtype=int)
    resized_img = np.empty((args.frame_size, args.image_size, args.image_size, 3))

    for j, img in enumerate(img_frame):
        resized_img[j, ] = cv2.resize(img, dsize=(args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)

    resized_img /= 255
    x[0,] = resized_img
    img_mask = np.all((resized_img == 0), axis=1)
    img_mask = np.all((img_mask == True), axis=1)
    img_mask = np.all((img_mask == True), axis=1)
    img_mask = np.logical_not(img_mask)
    mask[0,] = img_mask

    checkpoint_path = 'ckpts'.format(args.name) + '/cp-0005.ckpt'
    model = create_model(args)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(),
                  metrics=['accuracy', tf.keras.metrics.Precision()])
    model.load_weights(checkpoint_path)

    dataset = tf.data.Dataset.from_tensor_slices({'img': x, 'mask': mask})
    prob = model.predict_on_batch([x, mask])
    prob_class = np.argmax(prob)
