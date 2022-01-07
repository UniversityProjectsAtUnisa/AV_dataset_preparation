from os import path
from config import OUTPUT_DIR, TEMP_DIR
import h5py
from split_labels_by_class import LABEL_NAMES
import numpy as np
import random
import math
import argparse
from tqdm import tqdm


def validate_b_ratio(b_ratio):
    if not isinstance(b_ratio, str):
        raise TypeError("b_ratio must be of type string")
    if ":" not in b_ratio:
        raise ValueError("b_ratio must contain ':' (colon) character")
    splitted = b_ratio.split(":")
    if not any([s == "1" for s in splitted]):
        raise ValueError("b_ratio must contain at least one 1")
    return float(splitted[0])/float(splitted[1])


def delete_stepped_array(arr, amount):
    step_size = len(arr) / amount

    i = len(arr)-1
    while i >= 0:
        randomized_index = round(
            i) - random.randint(0, min(math.floor(step_size-1), math.floor(i)))
        arr = np.delete(arr, randomized_index)
        i -= step_size

    return arr


def shuffle_rowwise(arr):
    for i in range(len(arr)):
        np.random.shuffle(arr[i])
    return arr


def main():
    parser = argparse.ArgumentParser(description="Balance train datasets")
    parser.add_argument("--ids_path", type=str, default=path.join(TEMP_DIR, "ds.h5"),
                        help="h5 file with dataset")
    parser.add_argument("--dataset_path", type=str, default=path.join(OUTPUT_DIR, "balanced_ds.h5"),
                        help="h5 File that will contain the balanced dataset")
    parser.add_argument("--b_ratio", type=str, default="1:1",
                        help="Balance ratio common:rare class. Ex: 1:2 1:1 5:1")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed value")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not path.isfile(args.ids_path):
        raise Exception(f"ids_path {args.ids_path} not found")

    print("Validating balance ratio")
    b_ratio = validate_b_ratio(args.b_ratio)

    print("Initializing dataset file")
    h5py.File(args.dataset_path, "w").close()

    print("Copying test sets")
    for label in LABEL_NAMES:
        with h5py.File(args.ids_path) as f:
            x_test = f[f"{label}/x/test"][:]
            y_test = f[f"{label}/y/test"][:]

        with h5py.File(args.dataset_path, "a") as f:
            f.create_dataset(f"{label}/x/test", data=x_test)
            f.create_dataset(f"{label}/y/test", data=y_test)

    print("Copying validation sets")
    for label in LABEL_NAMES:
        with h5py.File(args.ids_path) as f:
            if f"{label}/x/val" not in f:
                continue
            x_val = f[f"{label}/x/val"][:]
            y_val = f[f"{label}/y/val"][:]

        with h5py.File(args.dataset_path, "a") as f:
            f.create_dataset(f"{label}/x/val", data=x_val)
            f.create_dataset(f"{label}/y/val", data=y_val)

    for label in LABEL_NAMES:
        print(f"Reading train labels {label}")
        with h5py.File(args.ids_path) as f:
            y_train = f[f"{label}/y/train"][:]

        one_count = np.unique(y_train, return_counts=True)[1][1]

        zeros = np.arange(0, len(y_train)-one_count)
        ones = np.arange(len(y_train) - one_count, len(y_train))
        y_train = None

        zeros_per_ds = math.floor(b_ratio*one_count)
        amount_to_delete = len(zeros) % zeros_per_ds
        zeros = delete_stepped_array(zeros, amount_to_delete)

        # Asserting deletion has worked
        assert len(zeros) % zeros_per_ds == 0

        zeros = zeros.reshape((zeros_per_ds, len(zeros) // zeros_per_ds))
        zeros = shuffle_rowwise(zeros).T

        y_train = np.concatenate([np.zeros(zeros_per_ds), np.ones(one_count)])
        print(f"Saving train labels {label}")
        with h5py.File(args.dataset_path, "a") as f:
            f.create_dataset(f"{label}/y/train", data=y_train)

        with h5py.File(args.ids_path) as f:

            x_train = f[f"{label}/x/train"]
            for classifier_index, row in enumerate(tqdm(zeros)):
                sample_indices = np.concatenate([row, ones])
                samples = x_train[sample_indices]
                with h5py.File(args.dataset_path, "a") as f:
                    f.create_dataset(
                        f"{label}/x/train/{classifier_index}", data=samples)
            print(f"Saved {len(zeros)} train datasets for {label}")

    print('Completed!')


if __name__ == '__main__':
    main()
