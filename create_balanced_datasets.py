import utils
import h5py
from split_labels_by_class import LABEL_NAMES
import numpy as np
import random
import math

SEED = 1
random.seed(SEED)
np.random.seed(SEED)


def delete_stepped_array(arr, amount):
    step_size = math.ceil(len(arr) / amount)

    for i in range(len(arr), 0, -step_size):
        randomized_index = i - random.randint(0, step_size-1)
        arr = np.delete(arr, randomized_index)

    return arr


def shuffle_rowwise(arr):
    for i in range(len(arr)):
        np.random.shuffle(arr[i])
    return arr


def main():
    args = utils.init_parameter(description='Dataset creator',
                                add_gt_arg=True, add_dataset_arg=True)

    print("Initializing dataset file")
    h5py.File(args.dataset_path, "w").close()

    print("Copying test sets")
    for label in LABEL_NAMES:
        with h5py.File(args.gt_path) as f:
            x_test = f[f"{label}/x/test"][:]
            y_test = f[f"{label}/y/test"][:]

        with h5py.File(args.dataset_path, "a") as f:
            f.create_dataset(f"{label}/x/test", data=x_test)
            f.create_dataset(f"{label}/y/test", data=y_test)

    for label in LABEL_NAMES:
        print(f"Reading train labels {label}")
        with h5py.File(args.gt_path) as f:
            y_train = f[f"{label}/y/train"][:]

        one_count = np.unique(y_train, return_counts=True)[1][1]

        zeros = np.arange(0, len(y_train)-one_count)
        ones = np.arange(len(y_train) - one_count, len(y_train))
        y_train = None

        amount_to_delete = len(zeros) % one_count
        zeros = delete_stepped_array(zeros, amount_to_delete)

        # Asserting deletion has worked
        assert len(zeros) % one_count == 0
        
        zeros = zeros.reshape((one_count, len(zeros) // one_count))
        zeros = shuffle_rowwise(zeros).T

        y_train = np.concatenate([np.zeros(one_count), np.ones(one_count)])
        print(f"Saving train labels {label}")
        with h5py.File(args.dataset_path, "a") as f:
            f.create_dataset(f"{label}/y/train", data=y_train)


        with h5py.File(args.gt_path) as f:

            x_train = f[f"{label}/x/train"]
            for classifier_index, row in enumerate(zeros):
                sample_indices = np.concatenate([row, ones])
                samples = x_train[sample_indices]
                print(f"Saving train images {label} classifier {classifier_index}")
                with h5py.File(args.dataset_path, "a") as f:
                    f.create_dataset(
                        f"{label}/x/train/{classifier_index}", data=samples)

    print('Completed!')


if __name__ == '__main__':
    main()
