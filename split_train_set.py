import h5py
import numpy as np
import random
import math

random.seed(1)

SPLIT_RATE = 0.1


def remove_datasets(f, *datasets):
    for dt in datasets:
        if dt in f:
            del f[dt]


def pop_stepped(array, percentage):
    step_size = math.ceil(1/percentage)

    popped = []
    for i in range(len(array), 0, -step_size):
        random_index = i - random.randint(0, step_size-1)
        popped.append(array[random_index])
        array = np.delete(array, random_index)
    return np.array(popped), array


if __name__ == '__main__':
    with h5py.File('train.h5', 'r') as f:
        x, y = f['x'][:], f['y'][:]

    y_train = {}

    y_train["y_beard"] = y[:, 0]  # TODO create dataset before this
    y_train["y_moustache"] = y[:, 1]
    y_train["y_glasses"] = y[:, 2]

    y_test = {}
    for k in y_train:
        y_test[k], y_train[k] = pop_stepped(y_train[k], SPLIT_RATE)

    all_dataset_names = [
        f"{prefix}/{k}" for prefix in ["test", "train"] for k in y_train]

    print(all_dataset_names)
    # with h5py.File('data/dt.h5', 'a') as f:
    #     # Remove old y data
    #     remove_datasets(f, *all_dataset_names)
    #     # Save new y data

    #     for k in y_train:
    #         f.create_dataset(f"train/{k}", data=y_train[k])
    #         f.create_dataset(f"test/{k}", data=y_test[k])
