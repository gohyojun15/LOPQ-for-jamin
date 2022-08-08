from omegaconf import DictConfig

from continual_helper.exp_replay import Buffer
from product_quantizer.utils import write_accuracy_csv
from .model_utils import get_model_obj
from .online_trainer import BiEncoderOnlineTrainer
import torch
from typing import List, Tuple
from torch import Tensor as T
from torch.utils.data import DataLoader


class BiEncoderUpperboundTrainer(BiEncoderOnlineTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def load_checkpoint_iter(self, iteration):
        model_to_load = get_model_obj(self.biencoder)
        print("load saved checkpoint")
        state_dict = torch.load(f"pretrained_epochiter{iteration}")
        model_to_load.load_state_dict(state_dict["model_dict"], strict=True)

    def run_online_continual_learning(self):
        # load models
        self.load_checkpoint_iter(iteration=0)
        self.optimizer = self.init_optimizer()

        online_train_datasets = self.get_online_train_dataset()
        online_eval_datasets = self.get_online_eval_dataset()
        _, pretrain_loader = self.pretrain_dataloader(eval=True)

        self.biencoder.eval()
        self.biencoder.cuda()

        """
        Generate Database with Product Quantized embeddings
        """
        pre_ctx_vectors, pre_passages, pre_passage_ids = self.extract_ctx_embeddings(pretrain_loader)
        pre_eval_loader = self.make_eval_loader(online_eval_datasets[0])
        pre_eval_ctx_vectors, pre_eval_passages, pre_eval_passage_ids = self.extract_ctx_embeddings(pre_eval_loader)

        # concat
        pre_ctx_vectors = torch.cat([pre_ctx_vectors, pre_eval_ctx_vectors], dim=0)
        pre_passages.extend(pre_eval_passages)
        pre_passage_ids.extend(pre_eval_passage_ids)

        # Product quantize DB.
        self.PQ.initialize(pre_ctx_vectors, passages=pre_passages, passage_ids=pre_passage_ids)
        """
        Do continual rounds
        """
        accuracy_dict = dict()
        for continual_round in range(int(self.cfg.num_continual_round)):
            for i, online_eval_data in enumerate(online_eval_datasets):
                accuracy = self.evaluate_retrieval_on_dataset(online_eval_data, self.cfg.online.evaluation.topk)
                accuracy_dict.setdefault(f"q{i}_recall1", []).append(accuracy[0])
                accuracy_dict.setdefault(f"q{i}_recall10", []).append(accuracy[9])
                accuracy_dict.setdefault(f"q{i}_recall50", []).append(accuracy[49])
            print(accuracy_dict)

            self.load_checkpoint_iter(iteration=continual_round + 1)

            # Reindex database.
            all_dataset = online_train_datasets[: continual_round + 1]
            all_dataset.extend(online_eval_datasets[: continual_round + 2])
            all_loader = [pretrain_loader]

            for d in all_dataset:
                dl = self.make_eval_loader(d)
                all_loader.append(dl)
            # extract database
            ctx_vec, passage_list, passage_id_list = self.extract_ctx_embeddings_multi_dataloader(all_loader)
            self.PQ.initialize(ctx_vec, passages=passage_list, passage_ids=passage_id_list)

        for i, online_eval_data in enumerate(online_eval_datasets):
            accuracy = self.evaluate_retrieval_on_dataset(online_eval_data, self.cfg.online.evaluation.topk)
            accuracy_dict.setdefault(f"q{i}_recall1", []).append(accuracy[0])
            accuracy_dict.setdefault(f"q{i}_recall10", []).append(accuracy[9])
            accuracy_dict.setdefault(f"q{i}_recall50", []).append(accuracy[49])
        print(accuracy_dict)
        write_accuracy_csv(accuracy_dict, exp_name=self.cfg.exp_all.exp_name)

    def extract_ctx_embeddings_multi_dataloader(self, dataloader_list: List[DataLoader]) -> (
    T, List[Tuple[str, str]], List[str]):
        for i, dataloader in enumerate(dataloader_list):
            if i == 0:
                ctx_vec, passage_list, passage_id_list = self.extract_ctx_embeddings(dataloader)
            else:
                _ctx_vec, _passage_list, _passage_id_list = self.extract_ctx_embeddings(dataloader)
                ctx_vec = torch.cat([ctx_vec, _ctx_vec], dim=0)
                passage_list.extend(_passage_list)
                passage_id_list.extend(_passage_id_list)
        return ctx_vec, passage_list, passage_id_list
