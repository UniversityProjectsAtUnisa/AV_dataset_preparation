import csv
from collections import defaultdict
import json
import argparse

LABEL_NAMES = ["beard", "moustache", "glasses"]


def main():
    parser = argparse.ArgumentParser(description="Index labels by class")
    parser.add_argument("--igt_path", type=str, default="ordered_labels.csv",
                        help="CSV File with ordered GroundTruth")
    parser.add_argument("--ogt_path", type=str, default="labels.json",
                        help="JSON File that will contain GroundTruth indexed by class")
    args = parser.parse_args()

    labels = defaultdict(list)
    with open(args.igt_path) as csv_file:
        data = csv.reader(csv_file, delimiter=",")
        for im_name, *res in data:
            for i, l in enumerate(LABEL_NAMES):
                prefix = "no-" if res[i] == "0" else ""
                labels[f"{prefix}{l}"].append(im_name)

    with open(args.ogt_path, "w") as f:
        json.dump(labels, f)


if __name__ == '__main__':
    main()
