from omegaconf import OmegaConf
import torch
from models.online_noupdate_lowerbound import NoUpdateLowerBound
from pytorch_lightning import seed_everything

def train_all(cfg):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_everything(cfg.exp_all.seed)
    # online round configuration.
    trainer = NoUpdateLowerBound(cfg)
    trainer.run_online_continual_learning()

if __name__=="__main__":
    cfg = OmegaConf.load("configure/text/noupdate_lowerbound.yaml")
    train_all(cfg)