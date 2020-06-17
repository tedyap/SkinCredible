import unittest
import numpy as np
from api.app import app
from model.opts import configure_args

app.testing = True
np.random.seed(0)
args = configure_args()
input_shape = (args.frame_size, args.image_size, args.image_size, 3)


class APITests(unittest.TestCase):
    """
    Unit tests to check if API can handle images of different frame size and image size
    """
    def test_equal_frame_size(self):
        with app.test_client() as client:
            img_frame = np.random.randint(0, 255, size=(args.frame_size, args.image_size, args.image_size, 3))
            rv = client.post('/predict', json={'img': img_frame.tolist()})
            json_data = rv.get_json()
            self.assertTrue(json_data['prob_class'] == 1)

    def test_small_frame_size(self):
        with app.test_client() as client:
            img_frame = np.random.randint(0, 255, size=(args.frame_size - 1, args.image_size, args.image_size, 3))
            rv = client.post('/predict', json={'img': img_frame.tolist()})
            json_data = rv.get_json()
            self.assertTrue(json_data['prob_class'] == 1)

    def test_large_frame_size(self):
        with app.test_client() as client:
            img_frame = np.random.randint(0, 255, size=(args.frame_size + 1, args.image_size, args.image_size, 3))
            rv = client.post('/predict', json={'img': img_frame.tolist()})
            json_data = rv.get_json()
            self.assertTrue(json_data['prob_class'] == 1)

    def test_large_image_size(self):
        with app.test_client() as client:
            img_frame = np.random.randint(0, 255, size=(args.frame_size, args.image_size + 1, args.image_size + 1, 3))
            rv = client.post('/predict', json={'img': img_frame.tolist()})
            json_data = rv.get_json()
            self.assertTrue(json_data['prob_class'] == 1)

    def test_small_image_size(self):
        with app.test_client() as client:
            img_frame = np.random.randint(0, 255, size=(args.frame_size, args.image_size - 1, args.image_size - 1, 3))
            rv = client.post('/predict', json={'img': img_frame.tolist()})
            json_data = rv.get_json()
            self.assertTrue(json_data['prob_class'] == 1)
