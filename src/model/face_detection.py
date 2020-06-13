import os
from io import BytesIO
import boto3
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import numpy as np
import logging
from opts import configure_args
from utils import rotate_image, set_logger
import s3fs
import pickle
import json
from itertools import islice


# extract a single face from a given image
def extract_face(filename, required_size=(160, 160)):
    image = rotate_image(filename)
    image = image.convert('RGB')
    pixels = asarray(image)

    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None
    else:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        return image


if __name__ == "__main__":

    args = configure_args()
    if not os.path.exists(os.path.join(args.model_dir, 'output')):
        os.makedirs('output')
    set_logger(os.path.join(args.model_dir, 'output/train.log'))

    fs = s3fs.S3FileSystem()
    s3 = boto3.client('s3')
    logging.info('Detecting faces...')

    with open(os.path.join(args.model_dir, 'data/user_data.txt'), 'r') as f:
        logging.info('Start...')
        for i, line in islice(enumerate(f), args.start, args.end):
            info = json.loads(line)
            user_id = int(info[0])
            user_img = info[3:-1]

            # data.shape = (frame, width, height, channel)
            data = np.zeros((args.frame_size, 160, 160, 3))
            count = 0
            logging.info('Extract face...')
            for img_path in user_img:
                with fs.open('s3://cureskin-dataset/images/{}'.format(img_path)) as file:
                    face = extract_face(file)
                    if face:
                        pixels = asarray(face)
                        data[count, :, :, :] = pixels
                        count += 1
            logging.info('Upload...')
            io = BytesIO()
            pickle.dump(data, io)
            io.seek(0)
            s3.upload_fileobj(io, 'cureskin-dataset', 'data/image_{}.pkl'.format(user_id))




