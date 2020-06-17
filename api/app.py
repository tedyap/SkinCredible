import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from model.network_architecture import create_model
from model.utils import pre_process
from model.opts import configure_args

app = Flask(__name__)
args = configure_args()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_frame = np.array(data['img'])
    x, mask = pre_process(img_frame, args)
    model = load_model()
    prob = model.predict_on_batch([x, mask])
    prob_class = np.argmax(prob)
    return jsonify(prob_class=int(prob_class))


def load_model():
    checkpoint_path = 's3://cureskin-dataset/ckpts/cp-0005.ckpt'
    model = create_model(args)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(),
                  metrics=['accuracy', tf.keras.metrics.Precision()])
    model.load_weights(checkpoint_path)
    return model
