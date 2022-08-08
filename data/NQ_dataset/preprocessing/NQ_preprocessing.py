"""
Preprocessing NQ_models dataset for lifelong setting.
make 1 + T splits. (1st pretraining, 2~T -> online continual).
"""
# Fill here
# Total Dataset Size: 65395
# Data path
NQ_DATA_PATH = "/Users/hyojungo/PycharmProjects/LifelongPQ/text/downloads/data/retriever"
# 1st pretraining dataset config
PRE_TRAIN_SIZE = 50095
PRE_TEST_SIZE = 300
# Online Continual Dataset (T ROUND)
NUM_ROUND = 5
ROUND_TRAIN = 2700
ROUND_TEST = 300
# Output Path
NQ_PROCESSED_ROOT = "PREPROCESS_DATA"
NQ_TRAIN_PATH = NQ_DATA_PATH + "/nq-train.json"
NQ_DEV_PATH = NQ_DATA_PATH + "/nq-dev.json"
SEED = 2022




import json
from typing import List
from pytorch_lightning import seed_everything
import random
import os
from pathlib import Path

seed_everything(2022)
def read_data_from_json_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            print("Reading file %s" % path)
            data = json.load(f)
            results.extend(data)
            print("Aggregated data size: {}".format(len(results)))
    return results

def save_data(paths, data):
    os.makedirs(Path(paths).parent, exist_ok=True)
    with open(paths, "w") as f:
        json.dump(data, f, indent=4)

all_datas = read_data_from_json_files([NQ_TRAIN_PATH, NQ_DEV_PATH])
root_name = NQ_PROCESSED_ROOT + f"/PRE{PRE_TRAIN_SIZE}_ROUND{NUM_ROUND}_TRAIN{ROUND_TRAIN}_TEST{ROUND_TEST}"

# shuffle all datas
random.shuffle(all_datas)

iter_pointer = 0
# PreTraining Data
pretrain_data_json = all_datas[iter_pointer : iter_pointer + PRE_TRAIN_SIZE]
iter_pointer += PRE_TRAIN_SIZE
# 1-st evaluation
pretrain_eval_data_json = all_datas[iter_pointer : iter_pointer + PRE_TEST_SIZE]
iter_pointer += PRE_TEST_SIZE

# T task Data
task_lists_train_datas = []
task_lists_eval_datas = []

for i in range(NUM_ROUND):
    train_data_json = all_datas[iter_pointer : iter_pointer + ROUND_TRAIN]
    iter_pointer += ROUND_TRAIN

    test_data_json = all_datas[iter_pointer : iter_pointer + ROUND_TEST]
    iter_pointer += ROUND_TEST

    task_lists_train_datas.append(train_data_json)
    task_lists_eval_datas.append(test_data_json)

# Save Datas
save_data(root_name + "/pretrain/train.json", pretrain_data_json)
save_data(root_name + "/pretrain/eval.json", pretrain_eval_data_json)
print(f"pretrain_size: {len(pretrain_data_json)}")
print(f"pretrain eval_size: {len(pretrain_eval_data_json)}")

for i in range(NUM_ROUND):
    save_data(root_name + f"/online/{i}task.json", task_lists_train_datas[i])
    save_data(root_name + f"/online/{i}eval.json", task_lists_eval_datas[i])
    print(f"online_train_size: {len(task_lists_train_datas[i])}")
    print(f"online_eval_size: {len(task_lists_eval_datas[i])}")
