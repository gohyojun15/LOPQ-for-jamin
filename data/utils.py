import os
import pathlib
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from .beir_dataset import JsonQADataset
from tqdm.autonotebook import tqdm
import json
import csv

class BeirLoader(GenericDataLoader):
    def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl",
                 qrels_folder: str = "qrels", qrels_file: str = "", data_name: str = None):
        super().__init__(data_folder, prefix, corpus_file, query_file, qrels_folder, qrels_file)
        self.data_name = data_name

    def _load_corpus(self):
        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[self.data_name + line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }

    def _load_queries(self):

        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[self.data_name + line.get("_id")] = line.get("text")

    def _load_qrels(self):
        reader = csv.reader(open(self.qrels_file, encoding="utf-8"),
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)

        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            # add prefix for discriminating each datasets
            query_id = self.data_name + query_id
            corpus_id = self.data_name + corpus_id

            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score


def get_dataset_by_name(name:str, tokenizer):
    if name == "msmarco":
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(name)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = beir_util.download_and_unzip(url, out_dir)
        dl = BeirLoader(data_path, data_name=name)
        corpus, quries, qrels = dl.load(split="train")
        train_dataset = JsonQADataset(quries, qrels, corpus, tokenizer)

        dl.queries = {}
        dev_corpus, dev_queries, dev_qrels = dl.load(split="dev")
        # val_dataset = JsonQADataset(dev_queries, dev_qrels, dev_corpus, tokenizer)
    elif name == "nq":
        # train dataset
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format("nq-train")
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = beir_util.download_and_unzip(url, out_dir)

        dl = BeirLoader(data_path, data_name=name)
        corpus, quries, qrels = dl.load(split="train")
        train_dataset = JsonQADataset(quries, qrels, corpus, tokenizer)

        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format("nq")
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = beir_util.download_and_unzip(url, out_dir)
        dl = BeirLoader(data_path, data_name=name)
        corpus, dev_queries, dev_qrels = dl.load(split="test")
        # val_dataset = JsonQADataset(dev_queries, dev_qrels, corpus, tokenizer)
    elif name == "hotpotqa":
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(name)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = beir_util.download_and_unzip(url, out_dir)
        dl = BeirLoader(data_path, data_name=name)
        corpus, quries, qrels = dl.load(split="train")
        train_dataset = JsonQADataset(quries, qrels, corpus, tokenizer)

        dl.queries = {}
        dev_corpus, dev_queries, dev_qrels = dl.load(split="test")
        # val_dataset = JsonQADataset(dev_queries, dev_qrels, dev_corpus, tokenizer)
    elif name == "fiqa":
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(name)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = beir_util.download_and_unzip(url, out_dir)
        dl = BeirLoader(data_path, data_name=name)
        corpus, quries, qrels = dl.load(split="train")
        train_dataset = JsonQADataset(quries, qrels, corpus, tokenizer)

        dl.queries = {}
        dev_corpus, dev_queries, dev_qrels = dl.load(split="test")
        # val_dataset = JsonQADataset(dev_queries, dev_qrels, dev_corpus, tokenizer)

    elif name == "fever":
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(name)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = beir_util.download_and_unzip(url, out_dir)
        dl = BeirLoader(data_path, data_name=name)
        corpus, quries, qrels = dl.load(split="train")
        train_dataset = JsonQADataset(quries, qrels, corpus, tokenizer)

        dl.queries = {}
        dev_corpus, dev_queries, dev_qrels = dl.load(split="test")
        # val_dataset = JsonQADataset(dev_queries, dev_qrels, dev_corpus, tokenizer)

    elif name == "quora":
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(name)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = beir_util.download_and_unzip(url, out_dir)
        dl = BeirLoader(data_path, data_name=name)
        corpus, quries, qrels = dl.load(split="dev")
        train_dataset = JsonQADataset(quries, qrels, corpus, tokenizer)

        dl.queries = {}
        dev_corpus, dev_queries, dev_qrels = dl.load(split="test")
        # val_dataset = JsonQADataset(dev_queries, dev_qrels, dev_corpus, tokenizer)

    return train_dataset, dev_corpus, dev_queries, dev_qrels



