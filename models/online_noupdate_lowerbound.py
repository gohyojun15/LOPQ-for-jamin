import torch

from data.utils import get_dataset_by_name
from evaluation.corpus_extraction import extract_corpus_embeddings, load_tensor, pickle_load
from .NQ_models.trainer import BiEncoderTrainer
from omegaconf import DictConfig
from product_quantizer.online_PQ import OnlinePQ
from product_quantizer.online_OPQ import OnlineOPQ
import os

class NoUpdateLowerBound(BiEncoderTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if cfg.PQ.name == "onlinePQ":
            self.PQ = OnlinePQ(cfg.PQ)
        elif self.PQ.name == "onlineOPQ":
            self.PQ = OnlineOPQ(cfg.PQ)

    # def init_optimizer(self) -> torch.optim.Optimizer:
    #     optim_config = self.cfg.online.optimizers
    #     no_decay = ["bias", "LayerNorm.weight"]
    #     params = [
    #         {
    #             "params": [p for n, p in self.biencoder.named_parameters() if not any(nd in n for nd in no_decay)],
    #             "weight_decay": optim_config.weight_decay,
    #         },
    #         {
    #             "params": [p for n, p in self.biencoder.named_parameters() if any(nd in n for nd in no_decay)],
    #             "weight_decay": 0.0,
    #         },
    #     ]
    #     if optim_config.name == "SGD":
    #         return torch.optim.SGD(params, lr=optim_config.learning_rate)
    #     elif optim_config.name == "Adam":
    #         return torch.optim.Adam(params, lr=optim_config.learning_rate)
    #     elif optim_config.name == "AdamW":
    #         return torch.optim.AdamW(params, lr=optim_config.learning_rate)

    def run_online_continual_learning(self):
        self.load_checkpoint()
        self.optimizer = self.init_optimizer()

        self.biencoder.cuda()


        train_dataset, dev_corpus, dev_queries, dev_qrels = get_dataset_by_name("msmarco", tokenizer=self.tensorizer)
        if not os.path.exists("database.pt"):
            extract_corpus_embeddings(dev_corpus, self, self.cfg, tokenizer=self.tensorizer)

        corpus_embedding = load_tensor("database.pt")
        passage_list = pickle_load("passage_list.p")
        passage_id_list = pickle_load("passage_id_list.p")

        a =3





