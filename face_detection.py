from io import BytesIO
import boto3
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import numpy as np
from sklearn.model_selection import train_test_split
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
    set_logger('output/train.log')

    fs = s3fs.S3FileSystem()
    s3 = boto3.client('s3')

    label_dict = {}
    user_list = []
    label_list = []
    with open('data/user_data.txt', 'r') as f:
        for i, line in islice(enumerate(f), args.start, args.end):
            info = json.loads(line)
            user_id = int(info[0])
            label_dict[user_id] = int(info[1])
            user_list.append(int(info[0]))
            label_list.append(int(info[1]))
            user_img = info[3:10]

            # data.shape = (frame, width, height, channel)
            data = np.zeros((args.frame_size, args.image_size, args.image_size, 3))
            count = 0
            for img_path in user_img:
                with fs.open('s3://cureskin-dataset/images/{}'.format(img_path)) as file:
                    face = extract_face(file)
                    if face:
                        pixels = asarray(face)
                        data[count, :, :, :] = pixels
                        count += 1

            io = BytesIO()
            pickle.dump(data, io)
            io.seek(0)
            s3.upload_fileobj(io, 'cureskin-dataset', 'data/image_{}.pkl'.format(user_id))

    x = np.array(user_list)
    y = np.array(label_list)

    # split into train and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=1)

    # split into train and validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.3, random_state=2)

    logging.info('Number of training samples: {}'.format(x_train.shape[0]))
    logging.info('Number of validation samples: {}'.format(x_val.shape[0]))
    logging.info('Number of testing samples: {}'.format(x_test.shape[0]))

    partition = {'train': x_train.tolist(), 'validation': x_val.tolist(), 'test': x_test.tolist()}

    with open('output/label_{}.json'.format(args.start), 'w') as f:
        json.dump(label_dict, f)

    with open('output/partition_{}.json'.format(args.start), 'w') as f:
        json.dump(partition, f)




