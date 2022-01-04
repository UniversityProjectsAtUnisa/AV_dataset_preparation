import csv
import functools

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

if __name__ == '__main__':
    labels = {}
    with open("labels.csv") as csv_file:
        data = csv.reader(csv_file, delimiter=",")
        for im_name, *res in data:
            labels[im_name] = im_name, *res

    ordered_labels = []

    keys = list(labels.keys())

    keys.sort(key=functools.cmp_to_key(cmp_images))

    for f in keys:
        ordered_labels.append(labels[f])

    with open("ordered_labels.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(ordered_labels)
