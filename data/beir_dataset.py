from .NQ_dataset.NQ_utils import Dataset, normalize_passage
import torch
import collections
from typing import Dict

BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])


class JsonQADataset(Dataset):
    def __init__(
            self,
            queries: Dict,
            qrels: Dict,
            corpus: Dict,
            tensorizer,
            special_token: str = None,
            normalize: bool = False,
    ):
        super().__init__(
            special_token=special_token
        )
        self.tensorizer = tensorizer
        self.queries = queries
        self.qrels = qrels
        self.corpus = corpus
        self.normalize = normalize

        self.query_ids = list(self.queries.keys())

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, index):
        query_id = self.query_ids[index]
        query = self.queries[query_id]
        query = self._process_query(query)

        for corpus_id, score in self.qrels[query_id].items():
            if score >= 1:
                positive_ctxs = self.corpus[corpus_id]
                break

        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                ctx["title"],
            )
        positive_passages = create_passage(positive_ctxs)
        # Tensorize query and passage
        query_tensor = self.tensorizer.tensorize_query(query)
        passage_tensor = self.tensorizer.tensorize_passage(positive_passages)

        return query_tensor, passage_tensor