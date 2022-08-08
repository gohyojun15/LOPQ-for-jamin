from omegaconf import OmegaConf
from models.NQ_models.online_trainer import BiEncoderOnlineTrainer
from models.NQ_models.fine_tune_trainer import BiEncoderFinetuneTrainer
from models.NQ_models.upper_bound_evaluator import BiEncoderUpperboundTrainer

import re
import pytorch_lightning as pl

def online_continual(cfg):
    pl.seed_everything(2023)
    # online round configuration.
    data_characteristic_str = str(cfg.data_path).split("/")[-1]
    numbers = re.findall(r"\d+", data_characteristic_str)

    cfg.num_train_data = numbers[0]
    cfg.num_continual_round = numbers[1]
    cfg.num_round_train = numbers[2]
    cfg.num_round_test = numbers[3]

    if cfg.exp_all.mode == "FineTune":
        trainer = BiEncoderFinetuneTrainer(cfg)
    elif cfg.exp_all.mode == "Continual":
        trainer = BiEncoderOnlineTrainer(cfg)
    elif cfg.exp_all.mode == "UpperBound":
        trainer = BiEncoderUpperboundTrainer(cfg)
    trainer.run_online_continual_learning()

if __name__=="__main__":
    cfg = OmegaConf.load("configure/text/online_pq.yaml")
    online_continual(cfg)