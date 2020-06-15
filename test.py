import pickle
import boto3
import cv2
import numpy as np
from matplotlib import pyplot as plt


s3 = boto3.client('s3')
response = s3.get_object(Bucket='cureskin-dataset', Key='new_data/image_{}.pkl'.format(10001))
body = response['Body'].read()
img_frame = pickle.loads(body)
x = np.empty((30, 80, 80, 3))
img_frame /= 255

print(img_frame.shape)


for i, img in enumerate(img_frame):
    x[i, ] = cv2.resize(img, dsize=(80, 80), interpolation=cv2.INTER_LINEAR)
    plt.imshow(x[i, ], interpolation='nearest')
    plt.show()

img_mask = np.all((img_frame == 0), axis=1)
img_mask = np.all((img_mask == True), axis=1)
img_mask = np.all((img_mask == True), axis=1)
img_mask = np.logical_not(img_mask)
print(img_mask)

print(x.shape)
