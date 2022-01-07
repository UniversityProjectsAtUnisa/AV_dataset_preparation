from collections import defaultdict
import os.path
from tqdm import tqdm

import h5py
import numpy as np
from cv2 import cv2
import json

import utils
from split_labels_by_class import LABEL_NAMES
import random
import math

import argparse


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
    parser = argparse.ArgumentParser(
        description="Create dataset file splitting data in train, test and possibly validation set")
    parser.add_argument("--igt_path", type=str, default="labels.json",
                        help="JSON File with GroundTruth indexed by class")
    parser.add_argument("--dataset_path", type=str, default="ds.h5",
                        help="h5 File that will contain the dataset")
    parser.add_argument("--im_path", type=str, required=True,
                        help="Path to images directory")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed value")
    parser.add_argument("--test_sr", type=float, default=0.1,
                        help="Test set split rate")
    parser.add_argument("--val_sr", type=float,
                        help="Validation set split rate")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    im_folder = args.im_path

    print("Reading whole data")
    train_names = {}
    with open(args.igt_path, mode='r') as f:
        train_names = json.load(f)

    print("Starting test stepped pop")
    test_names = defaultdict(list)
    for k, names in train_names.items():
        test_names[k] = pop_stepped(names, args.test_sr)

    val_names = None
    if args.val_sr is not None:
        print("Starting stepped pop")
        val_names = defaultdict(list)
        for k, names in train_names.items():
            val_names[k] = pop_stepped(names, args.val_sr)

    print("Initializing dataset file")
    h5py.File(args.dataset_path, "w").close()

    prefixes = ["no-", ""]
    sets = {"train": train_names, "test": test_names}
    if args.val_sr is not None:
        sets["val"] = val_names
    for label in LABEL_NAMES:
        for scope, s in sets.items():
            print(f"Retrieving images {label} {scope}")
            x, y = [], []
            for i, p in enumerate(prefixes):
                for im_name in tqdm(s[f"{p}{label}"]):
                    im = cv2.imread(os.path.join(im_folder, im_name))
                    if im.size == 0:
                        raise ValueError('Found image with size 0')
                    x.append(im)
                    y.append(i)

            print(f"Saving data {label} {scope}")
            with h5py.File(args.dataset_path, "a") as f:
                f.create_dataset(f"{label}/x/{scope}", data=np.array(x))
                f.create_dataset(f"{label}/y/{scope}", data=np.array(y))

    print('Completed!')


if __name__ == '__main__':
    main()
