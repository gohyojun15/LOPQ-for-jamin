DATASET_NAMES = ["msmarco", "nfcorpus", "nq", "hotpotqa", "fiqa", "fever", "nq-train", "quora"]
from beir import util

for dataset in DATASET_NAMES:
    print(dataset)
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, "datasets")

