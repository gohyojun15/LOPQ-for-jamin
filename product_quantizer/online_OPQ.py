import torch
from .product_quantization import OPQ
from .utils import duplicate_index_remove_passage
from operator import itemgetter
import numpy as np


class OnlineOPQ(OPQ):
    def __init__(self, opt):
        super().__init__(opt)
        self.key_list = []

    @torch.no_grad()
    def increase_counter(self, codes):
        B, M = codes.shape
        for b in range(B):
            code = codes[b]
            self.counters[range(M), code] += 1

    @torch.no_grad()
    def increase_db(self, codes, **kwargs):
        self.db = torch.cat((self.db, codes), dim=0)

        # increase DB.
        for key in self.key_list:
            value = getattr(self, key)
            aug_val = kwargs[key]
            if type(value) == list or type(value) == tuple:
                setattr(self, key, value + aug_val)

    @torch.no_grad()
    def initialize(self, vecs, **kwargs):
        """
        We follow Algorithm 1 in Online PQ
        """
        self.key_list = []
        # Exclude Duplicate passage(database components).
        remove_index = duplicate_index_remove_passage(passage_id_list=kwargs["passage_ids"])
        vecs = vecs[remove_index]
        for key, value in kwargs.items():
            if type(value) == list or type(value) == tuple:
                # slicing
                kwargs[key] = list(itemgetter(*remove_index)(value))
            elif type(value) == torch.Tensor:
                # slicing
                kwargs[key] = value[remove_index]

        # 1, 2 lines in online PQ
        self.fit(vecs)
        # 3 line in online PQ
        codes = self.encode(vecs)
        self.counters = torch.zeros((self.M, self.Ks), dtype=torch.int64)
        self.increase_counter(codes)

        self.db = codes
        # initialize save_components
        for key, value in kwargs.items():
            self.key_list.append(key)
            setattr(self, key, value)

    @torch.no_grad()
    def retrieval_topk(self, queries, k=50):
        # retrieve all
        if k == -1:
            k = self.db.shape[0]

        dtable = self.dtable(queries)
        # this code block uses GPU
        searched = dtable.adist(self.db)
        distances, index = torch.topk(searched, k=k, largest=False, sorted=True, dim=-1)
        # this code block uses GPU

        # indexing retrieval results.
        ret_dict = {}
        for key in self.key_list:
            variable = getattr(self, key)
            if type(variable) == list or type(variable) == tuple:
                ret_dict[key] = np.array(variable)[index.cpu()]
            elif type(variable) == torch.Tensor:
                ret_dict[key] = variable[index.cpu()]
        return distances, ret_dict

    @torch.no_grad()
    def handle_online_stream(self, vecs, **kwargs):
        codes = self.encode(vecs)
        self.increase_counter(codes)
        B, M = codes.shape
        code_residual = torch.zeros((self.M, self.Ks, self.Ds), dtype=torch.float32)
        codewords = self.decode(codes)
        # codewords update
        for b in range(B):
            code = codes[b]
            code_residual[range(M), code, :] += (vecs[b] - codewords[b]).view(M, self.Ds)
        # Save codewords
        pre_codewords = self.codewords.clone()
        self.codewords[torch.where(self.counters > 0)] += code_residual[torch.where(self.counters > 0)] * 1 / (
            self.counters[torch.where(self.counters > 0)].unsqueeze(dim=-1))

        # [Check every m]
        R_update_m = []
        for m in range(self.M):
            U, s, V = torch.linalg.svd(pre_codewords[m].T @ self.codewords[m])
            R_update_m.append(U @ V)
        update_R = torch.block_diag(*R_update_m)
        self.R = self.R @ update_R
        self.increase_db(codes, **kwargs)
