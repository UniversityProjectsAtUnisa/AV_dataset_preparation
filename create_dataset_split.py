from collections import defaultdict
import os.path
from tqdm import tqdm

import h5py
import numpy as np
from cv2 import cv2
import json

from split_labels_by_class import LABEL_NAMES
import random
import math

import argparse
from os import path
from config import OUTPUT_DIR, TEMP_DIR


def remove_datasets(f, *datasets):
    for dt in datasets:
        if dt in f:
            del f[dt]


def pop_stepped(l, percentage):
    step_size = math.ceil(1/percentage)

    p = []
    for i in range(len(l)-1, -1, -step_size):
        randomized_index = i - random.randint(0, step_size-1)
        if randomized_index >= 0:
            p.append(l.pop(randomized_index))
    return p


def init_args():
    default_separate_igt_path = path.join(TEMP_DIR, "labels.json")
    default_single_igt_path = path.join(TEMP_DIR, "joined_labels.json")
    default_separate_dataset_path = path.join(TEMP_DIR, "ds.h5")
    default_joined_dataset_path = path.join(OUTPUT_DIR, "test_ds.h5")
    parser = argparse.ArgumentParser(
        description="Create dataset file splitting data in train, test and possibly validation set")
    parser.add_argument("--igt_path", type=str, default=default_separate_igt_path,
                        help="JSON File with GroundTruth indexed by class")
    parser.add_argument("--ogt_path", type=str, default=path.join(TEMP_DIR, "unordered_train_labels.csv"),
                        help="Output CSV File where labels are saved if --joined_ds is given as input")
    parser.add_argument("--dataset_path", type=str, default=default_separate_dataset_path,
                        help="h5 File that will contain the dataset")
    parser.add_argument("--im_path", type=str, required=True,
                        help="Path to images directory")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed value")
    parser.add_argument("--test_sr", type=float, default=0.1,
                        help="Test set split rate")
    parser.add_argument("--val_sr", type=float,
                        help="Validation set split rate (on set without test, therefore if test_sr=val_sr, |val_sr| < |test_sr|)")
    parser.add_argument('--separate_ds', dest='separate_ds', action='store_true',
                        help="Datasets must be indexed by class (default)")
    parser.add_argument('--joined_ds', dest='separate_ds', action='store_false',
                        help="Datasets must not be indexed by class")
    parser.set_defaults(separate_ds=True)
    args = parser.parse_args()

    if args.igt_path == default_separate_igt_path and not args.separate_ds:
        args.igt_path = default_single_igt_path
    if args.dataset_path == default_separate_dataset_path and not args.separate_ds:
        args.dataset_path = default_joined_dataset_path
    return args


def main():
    args = init_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    im_folder = args.im_path

    if not path.isfile(args.igt_path):
        raise Exception(f"igt_path {args.igt_path} not found")

    if not path.isdir(im_folder):
        raise Exception(f"im_folder {im_folder} not found")

    print("Reading whole data")
    train_names = {}
    with open(args.igt_path) as f:
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

    if args.separate_ds:
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

    else:  # joined datasets
        sets = {"test": test_names}
        if args.val_sr is not None:
            sets["val"] = val_names

        for scope, s in sets.items():
            print(f"Retrieving images {scope}")
            x, y = [], []
            for k, im_names in s.items():
                print(f"Class {k}")
                for im_name in tqdm(im_names):
                    im = cv2.imread(os.path.join(im_folder, im_name))
                    if im.size == 0:
                        raise ValueError('Found image with size 0')
                    x.append(im)
                y.extend([np.array([int(c) for c in k])]*len(im_names))

            print(f"Saving data {scope}")
            with h5py.File(args.dataset_path, "a") as f:
                f.create_dataset(f"x/{scope}", data=np.array(x))
                f.create_dataset(f"y/{scope}", data=np.array(y))

        print("Saving training data on csv")
        train_labels = []
        for k, im_names in train_names.items():
            for im_name in im_names:
                im_name = f"{im_name},{','.join([c for c in k])}"
                train_labels.append(im_name)

        with open(args.ogt_path, "w") as f:
            f.writelines(l+'\n' for l in train_labels)

    print('Completed!')


if __name__ == '__main__':
    main()
