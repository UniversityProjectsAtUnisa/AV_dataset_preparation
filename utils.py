import argparse
import csv
from typing import Tuple


def init_parameter(description: str, add_gt_arg=False, add_res_arg=False, add_im_arg=False, add_dataset_arg=False,
                   add_model_arg=False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    if add_gt_arg:
        parser.add_argument("--gt_path", type=str, required=True, help="File CSV con la GroundTruth")
    if add_res_arg:
        parser.add_argument("--res_path", type=str, required=True, help="File CSV con i risultati")
    if add_im_arg:
        parser.add_argument("--im_path", type=str, required=True, help="Cartella delle immagini")
    if add_dataset_arg:
        parser.add_argument("--dataset_path", type=str, required=True, help="Dataset h5 file")
    if add_model_arg:
        parser.add_argument("--model_path", type=str, required=True, help="Percorso modello DCNN")
    return parser.parse_args()


def read_labels(path: str) -> Tuple[dict, dict, dict, int]:
    with open(path, mode='r') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        data_num = 0
        b_dict, m_dict, g_dict = {}, {}, {}
        for row in data:
            b_dict.update({row[0]: int(round(float(row[1])))})
            m_dict.update({row[0]: int(round(float(row[2])))})
            g_dict.update({row[0]: int(round(float(row[3])))})
            data_num += 1
        return b_dict, m_dict, g_dict, data_num