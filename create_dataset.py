import csv
import os.path

import h5py
import numpy as np
from cv2 import cv2

import utils


def main():
    args = utils.init_parameter(description='Dataset creator', add_gt_arg=True, add_im_arg=True, add_dataset_arg=True)
    im_folder = args.im_path

    print('Loading images and labels...')
    x, y = [], []
    with open(args.gt_path, mode='r') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        for im_name, *res in data:
            im = cv2.imread(os.path.join(im_folder, im_name))
            if im.size == 0:
                raise ValueError('Found image with size 0')
            x.append(im)
            y.append(tuple(int(r) for r in res))

    print('Saving h5 file...')
    with h5py.File(args.dataset_path, "w") as f:
        f.create_dataset('x', data=np.array(x))
        f.create_dataset('y', data=np.array(y))
    print('Completed!')


if __name__ == '__main__':
    main()
