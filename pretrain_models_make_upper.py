from omegaconf import OmegaConf

from data.utils import get_dataset_by_name
from models.NQ_models.online_trainer import BiEncoderOnlineTrainer
from models.NQ_models.trainer import BiEncoderTrainer
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import DataLoader, ConcatDataset

def pretrain(cfg):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_everything(cfg.exp_all.seed)

    trainer = BiEncoderTrainer(cfg)
    train_dataset, val_dataset, corpus = get_dataset_by_name(name="msmarco",tokenizer=trainer.tensorizer)

    # online_train_datasets = BiEncoderOnlineTrainer(cfg).get_online_train_dataset()
    print(train_dataset)
    # print(len(online_train_datasets))
    for i in range(0, 1):
        dataloader = DataLoader(
            dataset=ConcatDataset([train_dataset]),
            batch_size=cfg.train.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            num_workers=cfg.train.num_workers
        )
        trainer.run_pretrain(dataloader)
        trainer.save_checkpoint(epoch=f"iter{i}")


if __name__ == "__main__":
    cfg = OmegaConf.load("configure/text/noupdate_lowerbound.yaml")
    pretrain(cfg)