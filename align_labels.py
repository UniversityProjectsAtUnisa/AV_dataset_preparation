import csv
import functools
import argparse
from os import path
from config import TEMP_DIR, INPUT_DIR

NUMBER_OF_GROUPS = 25


def cmp_images(a, b):
    a_pieces = a.split("_")
    b_pieces = b.split("_")

    l = min(len(a_pieces), len(b_pieces))

    for i in range(l - 1):
        if int(a_pieces[i]) < int(b_pieces[i]):
            return -1
        elif int(a_pieces[i]) > int(b_pieces[i]):
            return 1

    if a_pieces[-1] < b_pieces[-1]:
        return -1
    elif a_pieces[-1] > b_pieces[-1]:
        return 1
    return 0


def init_args():
    parser = argparse.ArgumentParser(description="Order labels")
    parser.add_argument("--igt_path", type=str, default=path.join(INPUT_DIR, "labels.csv"),
                        help="CSV File with unordered GroundTruth")
    parser.add_argument("--ogt_path", type=str, default=path.join(TEMP_DIR, "ordered_labels.csv"),
                        help="CSV File that will contain the ordered GroundTruth")
    args = parser.parse_args()

    return args


def main():
    args = init_args()

    if not path.isfile(args.igt_path):
        raise Exception(f"igt_path {args.igt_path} not found")

    labels = {}
    with open(args.igt_path) as csv_file:
        data = csv.reader(csv_file, delimiter=",")
        for im_name, *res in data:
            labels[im_name] = im_name, *res

    ordered_labels = []

    keys = list(labels.keys())

    keys.sort(key=functools.cmp_to_key(cmp_images))

    for f in keys:
        ordered_labels.append(labels[f])

    with open(args.ogt_path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(ordered_labels)


if __name__ == '__main__':
    main()
