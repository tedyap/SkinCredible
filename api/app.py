import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
from model.network_architecture import create_model
from model.utils import pre_process
from model.opts import configure_args

import io
from PIL import Image


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret key'
app.config['SESSION_TYPE'] = 'filesystem'

img_dict = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist('file[]')
    resized_img = np.empty((args.frame_size, args.image_size, args.image_size, 3))

    if len(uploaded_files) == 1:
        return render_template('error.html', message='Please upload more than one selfies above!')
    elif not_allowed_file(uploaded_files):
        return render_template('error.html', message='Please upload jpg or jpeg files only.')
    else:
        for i, file in enumerate(uploaded_files):
            img_bytes = file.read()
            img = np.array(Image.open(io.BytesIO(img_bytes)))
            resized_img[i, ] = cv2.resize(img, dsize=(args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
        global img_dict
        img_dict['img'] = resized_img.tolist()
        return redirect(url_for('predict_as_get'))


@app.route('/predict', methods=['POST'])
def predict_as_post():
    data = request.get_json()
    img_frame = np.array(data['img'])
    x, mask = pre_process(img_frame, args)
    prob = model.predict_on_batch([x, mask])
    prob_class = np.argmax(prob)
    return jsonify(prob_class=int(prob_class))


@app.route('/predict', methods=['GET'])
def predict_as_get():
    global img_dict
    data = img_dict
    img_dict = {}
    img_frame = np.array(data['img'])
    x, mask = pre_process(img_frame, args)
    prob = model.predict_on_batch([x, mask])
    prob_class = np.argmax(prob)
    if prob_class == 1:
        return render_template('good.html')
    else:
        return render_template('bad.html')


def load_model():
    checkpoint_path = 's3://cureskin-dataset/ckpts/cp-0005.ckpt'
    model = create_model(args)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(),
                  metrics=['accuracy', tf.keras.metrics.Precision()])
    model.load_weights(checkpoint_path)
    return model


def not_allowed_file(uploaded_files):
    output = False
    print(uploaded_files)
    for file in uploaded_files:
        print(file.filename)
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in set(['jpg', 'jpeg'])):
            output = True
            break
    return output


if __name__ == '__main__':
    args = configure_args()
    model = load_model()
    app.run(host='0.0.0.0', port=80, threaded=False)