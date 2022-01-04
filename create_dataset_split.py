from collections import defaultdict
import os.path

import h5py
import numpy as np
from cv2 import cv2
import json

import utils
from split_labels_by_class import LABEL_NAMES
import random
import math

random.seed(1)
SPLIT_RATE = 0.1


def remove_datasets(f, *datasets):
    for dt in datasets:
        if dt in f:
            del f[dt]


def pop_stepped(l, percentage):
    step_size = math.ceil(1/percentage)

    p = []
    for i in range(len(l), 0, -step_size):
        randomized_index = i - random.randint(0, step_size-1)
        p.append(l.pop(randomized_index))
    return p


def main():
    args = utils.init_parameter(description='Dataset creator',
                                add_gt_arg=True, add_im_arg=True, add_dataset_arg=True)
    im_folder = args.im_path

    print("Reading whole data")
    train_names = {}
    with open(args.gt_path, mode='r') as f:
        train_names = json.load(f)

    print("Starting stepped pop")
    test_names = defaultdict(list)
    for k, names in train_names.items():
        test_names[k] = pop_stepped(names, SPLIT_RATE)

    print("Initializing dataset file")
    h5py.File(args.dataset_path, "w").close()

    prefixes = ["no-", ""]
    sets = {"train": train_names, "test": test_names}
    for label in LABEL_NAMES:
        for t, s in sets.items():
            print(f"Retrieving images {label} {t}")
            x, y = [], []
            for i, p in enumerate(prefixes):
                for im_name in s[f"{p}{label}"]:
                    im = cv2.imread(os.path.join(im_folder, im_name))
                    if im.size == 0:
                        raise ValueError('Found image with size 0')
                    x.append(im)
                    y.append(i)

            print(f"Saving data {label} {t}")
            with h5py.File(args.dataset_path, "a") as f:
                f.create_dataset(f"{label}/x/{t}", data=np.array(x))
                f.create_dataset(f"{label}/y/{t}", data=np.array(y))

    print('Completed!')


if __name__ == '__main__':
    main()
