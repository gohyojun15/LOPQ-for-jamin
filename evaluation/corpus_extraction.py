import torch
from multiprocessing.pool import Pool
from functools import partial
from torch.utils.data import Dataset, DataLoader
import collections
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from data.NQ_dataset.NQ_utils import normalize_passage
from torch.cuda.amp import autocast
import pickle
import numpy as np


def save_tensor(name, tensor):
    saves = tensor.numpy()
    with open(name, "wb") as f:
        np.save(f, saves)

def load_tensor(name):
    with open(name, "rb") as f:
        saves = np.load(f)
    return torch.from_numpy(saves)

def pickle_save(name, contents):
    with open(name, "wb") as f:
        pickle.dump(contents, f)

def pickle_load(name):
    with open(name, "rb") as f:
        return pickle.load(f)

BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])

class wrapper_for_dataparallel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.biencoder
        self.tensorizer = model.tensorizer


    def forward_passage(self, passage_tensor):
        ctx_attn_mask = self.tensorizer.get_attn_mask(passage_tensor)
        ctx_segments = torch.zeros_like(passage_tensor)
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.model.forward_ctx_model(passage_tensor, ctx_segments,
                                                                                 ctx_attn_mask,
                                                                                 representation_token_pos=0)
        return ctx_pooled_out

    def forward(self, x):
        return self.forward_passage(x)


class corpus_dataset(Dataset):
    def __init__(self, tokenizer, corpus):
        super().__init__()
        self.tensorizer = tokenizer
        self.corpus = corpus
        self.corpus_ids = list(self.corpus.keys())
        self.normalize = False

    def __len__(self):
        return len(self.corpus_ids)

    def __getitem__(self, index):
        corpus_id = self.corpus_ids[index]
        corpus = self.corpus[corpus_id]
        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                ctx["title"],
            )

        positive_corpus = create_passage(corpus)
        passage_tensor = self.tensorizer.tensorize_passage(positive_corpus)

        return corpus_id, positive_corpus, passage_tensor

def split(a, n):
    k, m = divmod(len(a), n)
    items = list(a.items())
    return [dict(items[i*k+min(i, m):(i+1)*k+min(i+1, m)]) for i in range(n)]


def extract_corpus_embeddings(corpus, model, cfg, tokenizer):
    print("extract embeddings")
    dp_model = nn.DataParallel(wrapper_for_dataparallel(model))
    dataset = corpus_dataset(tokenizer=tokenizer, corpus=corpus)
    loader = DataLoader(
        dataset, batch_size=cfg.online.retriev_batch_size, shuffle=False,
        num_workers=20,
    )

    ctx_vector_list = []
    passages_list = []
    passage_id_list = []
    for batch_sample in tqdm(loader,total=len(loader)):
        corpus_id_batch, corpus_batch, passage_tensor = batch_sample
        with torch.no_grad():
            tensor = passage_tensor.cuda()
            with autocast():
                embeeding = dp_model(tensor)
            ctx_vector = F.normalize(embeeding, dim=-1)
        ctx_vector_list.append(ctx_vector.cpu())
        passages_list.append(corpus_batch)
        passage_id_list.append(corpus_id_batch)

    ctx_vector_tensors = torch.cat(ctx_vector_list, dim=0)
    passages_list = [passage for bi_encoder_passages in passages_list for passage in
                     zip(bi_encoder_passages.title, bi_encoder_passages.text)]
    passage_id_list = [x_list for x in passage_id_list for x_list in x]

    save_tensor("database.pt", ctx_vector_tensors)
    pickle_save("passages_list.p", passages_list)
    pickle_save("passage_id_list.p", passage_id_list)

    return ctx_vector_tensors, passages_list, passage_id_list



