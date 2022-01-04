import os
import cv2
import argparse

import pandas as pd

NUMBER_OF_GROUPS = 25


if __name__ == '__main__':

    # Argument parser
    argpars = argparse.ArgumentParser()

    argpars.add_argument("--img-folder", required=True,
                         help="Path to the folder containing the images to label.")
    argpars.add_argument("--group-id", required=True, type=int,
                         help="ID of your group.")

    args = argpars.parse_args()

    # Get list of files in the folder
    if not os.path.exists(args.img_folder):
        raise RuntimeError(f"Path {args.img_folder} not exists!")

    file_list = os.listdir(args.img_folder)
    file_list.sort()
    file_list = [k for i,k in enumerate(file_list) if (i % NUMBER_OF_GROUPS == args.group_id - 1)]
    file_list = file_list[99: 196 + 1]

    # Check if the labels file exists
    if os.path.exists(f"group_{args.group_id}.csv"):
        print("Continuing existing annotation...")
        df = pd.read_csv(f"group_{args.group_id}.csv", index_col=False)

    else:
        print("New annotation...")
        data = {
            "file_name": file_list,
            "beard": [-1]*len(file_list),
            "moustache": [-1]*len(file_list),
            "glasses": [-1]*len(file_list)
        }

        df = pd.DataFrame(data=data)
    
    for row in df.iterrows():
        if row[1][1] == -1:
            try:
                img = cv2.imread(os.path.join(args.img_folder, row[1][0]))
                cv2.imshow("Current frame", img)
                cv2.waitKey(50)

                labels = [-1, -1, -1]

                while labels[0] not in [1, 0]:
                    try:
                        labels[0] = int(
                            input("Does the person has a beard? (0 = NO, 1 = YES) "))
                    except ValueError:
                        print("Please insert an integer value!")

                while labels[1] not in [1, 0]:
                    try:
                        labels[1] = int(
                            input("Does the person has moustache? (0 = NO, 1 = YES) "))
                    except ValueError:
                        print("Please insert an integer value!")

                while labels[2] not in [1, 0]:
                    try:
                        labels[2] = int(
                            input("Is the person wearing glasses? (0 = NO, 1 = YES) "))
                    except ValueError:
                        print("Please insert an integer value!")

                df.loc[row[0], ["beard"]] = labels[0]
                df.loc[row[0], ["moustache"]] = labels[1]
                df.loc[row[0], ["glasses"]] = labels[2]
                cv2.destroyAllWindows()
                cv2.waitKey(50)

            except KeyboardInterrupt:
                break
    
    df.to_csv(f"group_{args.group_id}.csv", index=False, header=False)
