from omegaconf import OmegaConf
from models.NQ_models.trainer import BiEncoderTrainer
from pytorch_lightning import seed_everything
import torch

def pretrain(cfg):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seed_everything(cfg.exp_all.seed)
    trainer = BiEncoderTrainer(cfg)
    trainer.run_pretrain()
    trainer.save_checkpoint()


if __name__ == "__main__":
    cfg = OmegaConf.load("configure/text/online_pq.yaml")
    pretrain(cfg)

    