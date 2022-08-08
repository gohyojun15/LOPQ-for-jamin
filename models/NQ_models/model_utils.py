from .bert_biencoder_wrapper import HFBertEncoder, get_bert_tokenizer, BertTensorizer
from .bi_encoder import BiEncoder
import torch.nn as nn

def get_bert_biencoder_components(cfg):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    question_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained
    )
    ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
    )
    biencoder = BiEncoder(question_encoder, ctx_encoder)
    tensorizer = get_bert_tensorizer(cfg)
    return biencoder, tensorizer


def get_bert_tensorizer(cfg):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg
    tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=cfg.encoder.do_lower_case)
    return BertTensorizer(tokenizer, sequence_length)


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model


