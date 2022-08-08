from evaluation.evaluation_utils import evaluate_results
from .trainer import BiEncoderTrainer
from omegaconf import DictConfig
import torch
from product_quantizer.online_PQ import OnlinePQ
from product_quantizer.online_OPQ import OnlineOPQ
from torch.utils.data import DataLoader
from typing import List, Tuple
import glob
from pathlib import Path
from data.NQ_dataset.biencoder_data import JsonQADataset
from torch import Tensor as T
import numpy as np
import torch.nn.functional as F
from operator import itemgetter

from product_quantizer.utils import write_accuracy_csv



class BiEncoderOnlineTrainer(BiEncoderTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if cfg.PQ.name == "onlinePQ":
            self.PQ = OnlinePQ(cfg.PQ)
        elif self.PQ.name == "onlineOPQ":
            self.PQ = OnlineOPQ(cfg.PQ)

    def init_optimizer(self) -> torch.optim.Optimizer:
        optim_config = self.cfg.online.optimizers
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in self.biencoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": optim_config.weight_decay,
            },
            {
                "params": [p for n, p in self.biencoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if optim_config.name == "SGD":
            return torch.optim.SGD(params, lr=optim_config.learning_rate)
        elif optim_config.name == "Adam":
            return torch.optim.Adam(params, lr=optim_config.learning_rate)
        elif optim_config.name == "AdamW":
            return torch.optim.AdamW(params, lr=optim_config.learning_rate)

    def run_online_continual_learning(self):
        # load models
        self.load_checkpoint()
        self.optimizer = self.init_optimizer()

        online_train_datasets = self.get_online_train_dataset()
        online_eval_datasets = self.get_online_eval_dataset()
        _, pretrain_loader = self.pretrain_dataloader(eval=True)

        """
        Generate Database with Product Quantized embeddings
        """
        self.biencoder.eval()
        self.biencoder.cuda()

        # Extract embeddings of train dataset and evaluation dataset.
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
            # Eval all dataset
            for i, online_eval_data in enumerate(online_eval_datasets):
                accuracy = self.evaluate_retrieval_on_dataset(online_eval_data, self.cfg.online.evaluation.topk)
                accuracy_dict.setdefault(f"q{i}_recall1", []).append(accuracy[0])
                accuracy_dict.setdefault(f"q{i}_recall10", []).append(accuracy[9])
                accuracy_dict.setdefault(f"q{i}_recall50", []).append(accuracy[49])

            # train and augment database.
            round_train_data = online_train_datasets[continual_round]
            round_eval_data = online_eval_datasets[continual_round + 1]
            round_train_loader = self.make_online_loader(round_train_data, train=True)
            round_eval_loader = self.make_online_loader(round_eval_data, train=False)

            round_train_loader.dataset.online_train = True
            round_eval_loader.dataset.extract_ctx_embedding = True

            for (train_batch, eval_batch) in zip(round_train_loader, round_eval_loader):
                train_query_tensors, train_passage_tensors, train_passage_ids, train_passages = train_batch
                eval_passage_tensors, eval_passage_ids, eval_passages = eval_batch

                self.biencoder.train()
                # train
                train_query_tensors = train_query_tensors.cuda()
                train_passage_tensors = train_passage_tensors.cuda()

                for _ in range(self.cfg.online.optim_num):
                    self.optimizer.zero_grad(set_to_none=True)
                    q_vectors, ctx_vectors = self.forward(train_query_tensors, train_passage_tensors)
                    loss = self.calc_in_negative_contrastive_loss(q_vectors, ctx_vectors)
                    loss.backward()
                    self.optimizer.step()

                # Data augment
                self.biencoder.eval()
                eval_passage_tensors = eval_passage_tensors.cuda()
                passage_tensors = torch.cat([train_passage_tensors, eval_passage_tensors], dim=0)
                train_passage_ids.extend(eval_passage_ids)
                passage_lists = [passage for bi_encoder_passages in [train_passages, eval_passages] for passage in
                 zip(bi_encoder_passages.title, bi_encoder_passages.text)]
                passage_ids = train_passage_ids

                # if database contains passages do not add it.
                alive_index = []
                passage_set = set(self.PQ.passage_ids)
                for index, passage_id in enumerate(passage_ids):
                    if not passage_id in passage_set:
                        alive_index.append(index)

                alive_index = list(set(alive_index))
                alive_index.sort()

                """
                Increase dataset
                """
                if len(alive_index) == 0:
                    continue
                elif len(alive_index) == 1:
                    aug_passage_tensors = passage_tensors[alive_index]
                    aug_passage_ids = [itemgetter(*alive_index)(passage_ids)]
                    aug_passage_lists = [itemgetter(*alive_index)(passage_lists)]
                else:
                    aug_passage_tensors = passage_tensors[alive_index]
                    aug_passage_ids = list(itemgetter(*alive_index)(passage_ids))
                    aug_passage_lists = list(itemgetter(*alive_index)(passage_lists))

                with torch.no_grad():
                    aug_passage_embeddings = self.forward_passage(aug_passage_tensors)
                    aug_passage_embeddings = F.normalize(aug_passage_embeddings, dim=-1)

                # increase database.
                self.PQ.handle_online_stream(
                    vecs=aug_passage_embeddings.cpu(),
                    passage_ids=aug_passage_ids,
                    passages=aug_passage_lists
                )
            round_train_loader.dataset.online_train = False
            round_eval_loader.dataset.extract_ctx_embedding = False

        for i, online_eval_data in enumerate(online_eval_datasets):
            accuracy = self.evaluate_retrieval_on_dataset(online_eval_data, self.cfg.online.evaluation.topk)
            accuracy_dict.setdefault(f"q{i}_recall1", []).append(accuracy[0])
            accuracy_dict.setdefault(f"q{i}_recall10", []).append(accuracy[9])
            accuracy_dict.setdefault(f"q{i}_recall50", []).append(accuracy[49])
        write_accuracy_csv(accuracy_dict, exp_name=self.cfg.exp_all.exp_name)


    def evaluate_retrieval_on_dataset(self, dataset, k=-1):
        eval_dataloader = self.make_eval_loader(dataset)
        dataset.eval = True # mark
        # retrieval.
        results = {
            "passages": [],
            "answers_list": []
        }
        for samples_batch in eval_dataloader:
            query_tensors, answer_list = samples_batch
            answer_list = [answer.split("/") for answer in answer_list]

            with torch.no_grad():
                q_vectors = self.forward_question(query_tensors.cuda())
                q_vectors = F.normalize(q_vectors, dim=-1)
            distances, retrieved_results = self.PQ.retrieval_topk(queries=q_vectors.cpu(), k=k)

            results["passages"].append(retrieved_results["passages"])
            results["answers_list"].append(answer_list)

        # evaluation.
        results["passages"] = np.concatenate(results["passages"], axis=0)
        results["answers_list"] = [i for x in results["answers_list"] for i in x]

        accuracy, all_results = evaluate_results(results)
        dataset.eval = False
        return accuracy

    def extract_ctx_embeddings(self, dataloader) -> (T, List[Tuple[str,str]], List[str]):
        """
        from dataloader, extract parts for constructing PQ database.
        """
        # dataset flag
        dataloader.dataset.extract_ctx_embedding = True

        ctx_vector_list = []
        passages_list = []
        passage_id_list = []

        for iterations, samples_batch in enumerate(dataloader):
            passage_tensors, passage_ids, passages = samples_batch
            with torch.no_grad():
                passage_tensors = passage_tensors.cuda()
                ctx_vector = self.forward_passage(passage_tensors)
                ctx_vector = F.normalize(ctx_vector, dim=-1)
            ctx_vector_list.append(ctx_vector.cpu())
            passages_list.append(passages)
            passage_id_list.append(passage_ids)

        ctx_vector_tensors = torch.cat(ctx_vector_list, dim=0)
        passages_list = [passage for bi_encoder_passages in passages_list for passage in zip(bi_encoder_passages.title, bi_encoder_passages.text)]
        passage_id_list = [x_list for x in passage_id_list for x_list in x]

        dataloader.dataset.extract_ctx_embedding = False
        return ctx_vector_tensors, passages_list, passage_id_list

    def make_online_loader(self, dataset, train=True):
        """
        make dataloader for online train in continual step.
        """
        if train:
            return DataLoader(
                dataset,
                batch_size=self.cfg.online.batch_size,
                shuffle=False,
                num_workers=self.cfg.train.num_workers,
                pin_memory=True,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.cfg.online.eval_batch_size,
                shuffle=False,
                num_workers=self.cfg.train.num_workers,
                pin_memory=True,
            )

    def make_eval_loader(self, dataset):
        """
        make evaluation loader: batch size is big,
        """
        return DataLoader(
            dataset,
            batch_size=self.cfg.online.retriev_batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def get_online_train_dataset(self) -> List[JsonQADataset]:
        data_path = Path(self.cfg.data_path) / "online"
        path_list = glob.glob(str(data_path / "*task.json"))
        path_list.sort()

        dataset_list = []
        for path in path_list:
            dataset = JsonQADataset(path, self.tensorizer)
            dataset.load_data()
            dataset_list.append(dataset)
        return dataset_list

    def get_online_eval_dataset(self) -> List[JsonQADataset]:
        path_list = [str(Path(self.cfg.data_path) / "pretrain" / "eval.json")]
        data_path = Path(self.cfg.data_path) / "online"
        tmp_list = glob.glob(str(data_path / "*eval.json"))
        tmp_list.sort()
        path_list.extend(tmp_list)

        dataset_list = []
        for path in path_list:
            dataset = JsonQADataset(path, self.tensorizer)
            dataset.load_data()
            dataset_list.append(dataset)
        return dataset_list