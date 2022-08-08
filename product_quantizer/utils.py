from collections import defaultdict
import csv
import os

def duplicate_index_remove_passage(passage_id_list):
    passage_uid = defaultdict(int)
    for i, passage_id in enumerate(passage_id_list):
        passage_uid[passage_id] = i
    removed_index = [value for key, value in passage_uid.items()]
    return removed_index


def write_accuracy_csv(accuracy_dicts, exp_name):
    file_path = "logs/" + exp_name + "/results.csv"
    if not os.path.exists("logs/" + exp_name):
        os.makedirs("logs/" + exp_name)

    with open(file_path, "w") as f:
        writer = csv.writer(f, accuracy_dicts.keys())
        for key, value in accuracy_dicts.items():
            row = [key]
            row.extend(value)
            writer.writerow(row)

def soft_quantization():

    return