import csv
from collections import defaultdict
import json
import argparse

LABEL_NAMES = ["beard", "moustache", "glasses"]


def init_args():
    default_separate_ogt_path = "labels.json"
    default_joined_ogt_path = "joined_labels.json"
    default_separate_igt_path = "train_labels.csv"
    default_joined_igt_path = "ordered_labels.csv"
    parser = argparse.ArgumentParser(description="Index labels by class")
    parser.add_argument("--igt_path", type=str, default=default_separate_igt_path,
                        help="CSV File with ordered GroundTruth")
    parser.add_argument("--ogt_path", type=str, default=default_separate_ogt_path,
                        help="JSON File that will contain GroundTruth indexed by class")
    parser.add_argument('--separate_ds', dest='separate_ds', action='store_true',
                        help="JSON File must be prepared for separate dataset (default)")
    parser.add_argument('--joined_ds', dest='separate_ds', action='store_false',
                        help="JSON File must be prepared for single dataset")
    parser.set_defaults(separate_ds=True)
    args = parser.parse_args()

    if args.ogt_path == default_separate_ogt_path and not args.separate_ds:
        args.ogt_path = default_joined_ogt_path
    if args.igt_path == default_separate_igt_path and not args.separate_ds:
        args.igt_path = default_joined_igt_path

    return args


def main():
    args = init_args()

    labels = defaultdict(list)
    with open(args.igt_path) as csv_file:
        data = csv.reader(csv_file, delimiter=",")
        for im_name, *res in data:
            if args.separate_ds:
                for i, l in enumerate(LABEL_NAMES):
                    prefix = "no-" if res[i] == "0" else ""
                    labels[f"{prefix}{l}"].append(im_name)

            else:  # single_ds
                res = "".join(res)
                labels[res].append(im_name)

    if args.separate_ds:
        labels["no-moustache"] = list(set(labels["no-moustache"]
                                          ) - set(labels["beard"]))

    with open(args.ogt_path, "w") as f:
        json.dump(labels, f)


if __name__ == '__main__':
    main()
