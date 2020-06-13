import os
import json
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from src.model.utils import set_logger
from opts import configure_args

if __name__ == "__main__":

    args =configure_args()

    set_logger(os.path.join(args.model_dir, 'output/train.log'))

    user_list = []
    label_list = []
    label_dict = {}
    with open(os.path.join(args.model_dir, 'data/user_data.txt')) as f:
        for line in f:
            data = json.loads(line)
            user_list.append(int(data[0]))
            label_list.append(int(data[1]))
            if int(data[1]) == 0:
                label_dict[int(data[0])] = [1, 0]
            else:
                label_dict[int(data[0])] = [0, 1]

        x = np.array(user_list)
        y = np.array(label_list)

        # split into train and test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.15, random_state=1)

        # split into train and validation set
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.15, random_state=2)

        (unique, counts) = np.unique(label_list, return_counts=True)
        frequencies = np.asarray((unique, counts)).T

        logging.info('Number of positive samples in total: {}'.format(frequencies[1][1]))
        logging.info('Number of negative samples in total: {}'.format(frequencies[0][1]))
        logging.info('Number of training samples: {}'.format(x_train.shape[0]))
        logging.info('Number of validation samples: {}'.format(x_val.shape[0]))
        logging.info('Number of testing samples: {}'.format(x_test.shape[0]))

        partition = {'train': x_train.tolist(), 'validation': x_val.tolist(), 'test': x_test.tolist()}

        with open(os.path.join(args.model_dir, 'data/label.json'), 'w') as f:
            json.dump(label_dict, f)

        with open(os.path.join(args.model_dir, 'data/partition.json'), 'w') as f:
            json.dump(partition, f)
