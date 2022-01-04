import os
import argparse

from tqdm import tqdm

if __name__ == '__main__':
    # Argument parser
    argpars = argparse.ArgumentParser()

    argpars.add_argument("--labels-folder", required=True,
                         help="Path to the folder containing the images to label.")

    args = argpars.parse_args()

    # Get list of files in the folder
    if not os.path.exists(args.labels_folder):
        raise RuntimeError(f"Path {args.labels_folder} not exists!")

    file_list = os.listdir(args.labels_folder)

    with open("labels.csv", 'w') as file:
        for g in tqdm(file_list):
            with open(os.path.join(args.labels_folder, g), 'r') as gt:
                file.writelines(gt.readlines())
