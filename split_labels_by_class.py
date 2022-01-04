import csv
from collections import defaultdict
import json

LABEL_NAMES = ["beard", "moustache", "glasses"]

if __name__ == '__main__':
    labels = defaultdict(list)
    with open("ordered_labels.csv") as csv_file:
        data = csv.reader(csv_file, delimiter=",")
        for im_name, *res in data:
            for i, l in enumerate(LABEL_NAMES):
                prefix = "no-" if res[i] == "0" else ""
                labels[f"{prefix}{l}"].append(im_name)

    with open("labels.json", "w") as f:
        json.dump(labels, f)
