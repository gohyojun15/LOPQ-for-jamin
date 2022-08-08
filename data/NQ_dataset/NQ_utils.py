import torch
import os
import glob
import json
from typing import List


def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question


def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    if ctx_text.startswith('"'):
        ctx_text = ctx_text[1:]
    if ctx_text.endswith('"'):
        ctx_text = ctx_text[:-1]
    return ctx_text

class Dataset(torch.utils.data.Dataset):
    """
    Super class for Dataset.
    """

    def __init__(
            self,
            special_token: str = None,
    ):
        self.special_token = special_token
        self.query_special_suffix = None
        self.data = []

    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        raise NotImplementedError

    def calc_total_data_len(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def _process_query(self, query: str):
        # as of now, always normalize query
        query = normalize_question(query)
        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix

        return query
