import os
import json
import pickle
import boto3
import cv2
from opts import configure_args
from io import BytesIO
import time

if __name__ == "__main__":
    args = configure_args()

    s3 = boto3.client('s3')
    with open(os.path.join(args.model_dir, 'data/label.json')) as f:
        label = json.load(f)

    for user_id, value in label.items():
        if value == [1, 0]:
            start = time.time()
            response = s3.get_object(Bucket='cureskin-dataset', Key='new_data/image_{}.pkl'.format(user_id))
            body = response['Body'].read()
            img_frame = pickle.loads(body)
            for i, img in enumerate(img_frame):
                img_frame[i, ] = cv2.flip(img, 1)

            with open(os.path.join(args.model_dir, 'data/user_data.txt'), 'a') as note_file:
                json.dump(['999' + str(user_id)] + [str(0)], note_file)
                note_file.write('\n')
            io = BytesIO()
            pickle.dump(img_frame, io)
            io.seek(0)
            s3.upload_fileobj(io, 'cureskin-dataset', 'new_data/image_#{}.pkl'.format(user_id))
            end = time.time()
            print(end - start)


