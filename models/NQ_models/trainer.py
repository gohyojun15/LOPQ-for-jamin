import collections

from omegaconf import DictConfig
from .model_utils import get_bert_biencoder_components, get_model_obj
import torch
from transformers import AdamW
from data.NQ_dataset.biencoder_data import JsonQADataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F
from tqdm import tqdm



class BiEncoderTrainer(object):
    def __init__(self, cfg: DictConfig):
        model, tensorizer = get_bert_biencoder_components(cfg)
        self.biencoder = model
        self.tensorizer = tensorizer
        self.cfg = cfg
        self.optimizer = None

    def run_pretrain(self, dataloader=None):
        self.optimizer = self.init_optimizer()
        if self.cfg.exp_all.amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()


        # Dataparallel
        self.biencoder = nn.DataParallel(self.biencoder.cuda(), device_ids=self.cfg.train.gpus)
        self.biencoder.train()

        for epoch in range(self.cfg.train.epoch):
            if epoch % 2 == 0:
                self.save_checkpoint(epoch)

            for iterations, samples_batch in tqdm(enumerate(dataloader)):
                self.optimizer.zero_grad(set_to_none=True)
                query_tenso, passage_tensors = samples_batch
                passage_tensors = passage_tensors.cuda(non_blocking=True)
                query_tensors = query_tensors.cuda(non_blocking=True)

                if self.cfg.exp_all.amp:
                    with autocast():
                        q_vectors, ctx_vectors = self.forward(query_tensors, passage_tensors)
                        loss = self.calc_in_negative_contrastive_loss(q_vectors, ctx_vectors)
                    # scale_before_step = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    scaler.step(self.optimizer)
                    scaler.update()

                else:
                    q_vectors, ctx_vectors = self.forward(query_tensors, passage_tensors)

                    loss = self.calc_in_negative_contrastive_loss(q_vectors, ctx_vectors)
                    loss.backward()
                    self.optimizer.step()

                if iterations % self.cfg.train.logging_freq == 0:
                    print(f"epoch : {epoch} iterations : {iterations}  loss : {loss.item()}")


    def init_optimizer(self) -> torch.optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in self.biencoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.train.weight_decay,
            },
            {
                "params": [p for n, p in self.biencoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params, lr=self.cfg.train.learning_rate, eps=self.cfg.train.adam_eps)
        return optimizer

    def calc_in_negative_contrastive_loss(self, q_vectors: T, ctx_vectors: T) -> T:
        q_vectors_norm = F.normalize(q_vectors, dim=-1)
        ctx_vectors_norm = F.normalize(ctx_vectors, dim=-1)
        scores = torch.matmul(q_vectors_norm, torch.transpose(ctx_vectors_norm, 0, 1))
        softmax_score = F.log_softmax(scores / self.cfg.train.loss.tau, dim=1)
        target = torch.tensor(range(q_vectors_norm.shape[0])).to(softmax_score.device)
        loss = F.nll_loss(softmax_score, target, reduction="mean")
        return loss

    def forward(self, question_tensor: T, passage_tensor: T):
        ctx_attn_mask = self.tensorizer.get_attn_mask(passage_tensor)
        ctx_segments = torch.zeros_like(passage_tensor)

        q_attn_mask = self.tensorizer.get_attn_mask(question_tensor)
        question_segments = torch.zeros_like(question_tensor)

        q_vectors, ctx_vectors = self.biencoder(
            question_tensor,
            question_segments,
            q_attn_mask,
            passage_tensor,
            ctx_segments,
            ctx_attn_mask,
            0
        )
        return q_vectors, ctx_vectors

    def forward_passage(self, passage_tensor: T):
        ctx_attn_mask = self.tensorizer.get_attn_mask(passage_tensor)
        ctx_segments = torch.zeros_like(passage_tensor)
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.biencoder.forward_ctx_model(passage_tensor, ctx_segments,
                                                                                 ctx_attn_mask,
                                                                                 representation_token_pos=0)
        return ctx_pooled_out

    def forward_question(self, question_tensor: T):
        q_attn_mask = self.tensorizer.get_attn_mask(question_tensor)
        question_segments = torch.zeros_like(question_tensor)
        _q_seq, q_pooled_out, _q_hidden = self.biencoder.forward_question_model(question_tensor, question_segments,
                                                                                q_attn_mask, representation_token_pos=0)
        return q_pooled_out

    def load_checkpoint(self):
        model_to_load = get_model_obj(self.biencoder)
        print("load saved checkpoint")
        state_dict = torch.load(self.cfg.save_path)
        model_to_load.load_state_dict(state_dict["model_dict"], strict=True)

    def save_checkpoint(self, epoch=None):
        cfg = self.cfg
        model_to_save = get_model_obj(self.biencoder)
        CheckpointState = collections.namedtuple(
            "CheckpointState",
            [
                "model_dict",
                "optimizer_dict",
                "params",
            ],
        )
        state = CheckpointState(
            model_to_save.state_dict(),
            self.optimizer.state_dict(),
            cfg
        )
        if epoch is not None:
            torch.save(state._asdict(), self.cfg.save_path + f"_epoch{epoch}")
            print(f"Saved checkpoint at {self.cfg.save_path}_epoch{epoch}")
        else:
            torch.save(state._asdict(), self.cfg.save_path)
            print(f"Saved checkpoint at {self.cfg.save_path}")
