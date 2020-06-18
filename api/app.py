import tensorflow as tf
from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
from model.network_architecture import create_model
from model.utils import pre_process, detect_face, pre_process_post
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
    print(uploaded_files)
    if len(uploaded_files) == 1:
        return render_template('error.html', message='Please upload more than one selfies above!')
    elif not_allowed_file(uploaded_files):
        return render_template('error.html', message='Please upload jpg or jpeg files only.')
    else:
        for i, file in enumerate(uploaded_files):
            img_bytes = file.read()
            img = np.array(Image.open(io.BytesIO(img_bytes)))
            resized_img[i, ] = detect_face(img, args)
        resized_img /= 255
        global img_dict
        img_dict['img'] = resized_img.tolist()
        return redirect(url_for('predict_as_get'))


@app.route('/predict', methods=['POST'])
def predict_as_post():
    args = configure_args()
    model = load_model(args)
    data = request.get_json()
    img_frame = np.array(data['img'])
    x, mask = pre_process_post(img_frame, args)
    prob = np.squeeze(model.predict_on_batch([x, mask]))
    prob_class = np.argmax(prob)
    return jsonify(prob_class=int(prob_class), negative_prob=float(prob[0]), positive_prob=float(prob[1]))


@app.route('/predict', methods=['GET'])
def predict_as_get():
    global img_dict
    data = img_dict
    img_dict = {}
    img_frame = np.array(data['img'])
    x, mask = pre_process(img_frame, args)

    if np.all((x == 0)):
        return render_template('error.html', message='SkinCredible did not find any faces...')

    prob = np.squeeze(model.predict_on_batch([x, mask]))
    prob_class = np.argmax(prob)
    if prob_class == 1:
        conf = prob[1] * 100
        message = 'SkinCredible is {:.2f}% confident that your facial skin condition has improved!'.format(conf)
        return render_template('good.html', message=message)
    else:
        conf = prob[0] * 100
        message = 'SkinCredible is {:.2f}% confident that your facial skin condition has become worse'.format(conf)
        return render_template('bad.html', message=message)


def load_model(args):
    checkpoint_path = 's3://cureskin-dataset/ckpts/cp-0005.ckpt'
    model = create_model(args)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(),
                  metrics=['accuracy', tf.keras.metrics.Precision()])
    model.load_weights(checkpoint_path)
    return model


def not_allowed_file(uploaded_files):
    output = False
    for file in uploaded_files:
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in set(['png', 'jpg', 'jpeg'])):
            output = True
            break
    return output


if __name__ == '__main__':
    args = configure_args()
    model = load_model(args)
    app.run(host='0.0.0.0', port=5000, threaded=False)
