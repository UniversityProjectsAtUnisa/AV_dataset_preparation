#!/bin/bash

python align_labels.py && \
python split_labels_by_class.py --joined_ds && \
python create_dataset_split.py --joined_ds --im_path utkface && \
python align_labels.py --igt_path tmp/unordered_train_labels.csv --ogt_path tmp/train_labels.csv && \
python split_labels_by_class.py && \
python create_dataset_split.py --val_sr 0.1 --im_path utkface && \
python create_balanced_datasets.py