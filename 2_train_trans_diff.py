import time
import torch

from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, ModelSummary

from utils.logging import Log
from utils import parse_config_from_args

from pathlib import Path
from model.TransformerDiffusion import TransDiffusionCombineModel
from model.TransformerDiffusion.dataloader import TransDiffusionDataset

import os
# Set WANDB_CACHE_DIR to a local directory to avoid no space left error
# cmd to clean up the cache: `wandb artifact cache cleanup 1GB`
os.environ['WANDB_CACHE_DIR'] = (Path() / 'wandb/cache').resolve().as_posix()
os.environ['WANDB_DATA_DIR'] = (Path() / 'wandb/data').resolve().as_posix()
import wandb

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    run_name = time.strftime("%m-%d-%I%p-%M-%S")
    config = parse_config_from_args()
    print(config['device'])
    use_wandb = config['wandb']['use']

    seed_everything(config['seed'])

    if use_wandb:
        wandb_run = wandb.init(
            config=config, project=config['wandb']['project'],
            entity=config['wandb']['entity'], name=run_name,
        )
        wandb_logger = WandbLogger()

    model = TransDiffusionCombineModel(config)

    # Configure data module
    d_configs = config['dataset_n_dataloader']
    train_dataloader = DataLoader(TransDiffusionDataset(
                dataset_path=d_configs['dataset_path'],
                description_for_each_file=d_configs['description_for_each_file'],
                cut_off=d_configs['cut_off'],
                enc_data_fieldname=d_configs['enc_data_fieldname']
            ),
            num_workers=d_configs['n_workers'], batch_size=d_configs['batch_size'],
            drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True)



    # Configure save checkpoint callback
    checkpoint_callback = ModelCheckpoint(
            every_n_train_steps=config['checkpoint']['freq'],
            dirpath=config['checkpoint']['path'] + '/' + run_name,
            filename="{epoch:04d}-{loss:.5f}",
        )

    # Configure trainer
    optional_kw_args = dict()
    if use_wandb:
        optional_kw_args['logger'] = wandb_logger

    trainer = Trainer(devices=config['devices'], accelerator=config["accelerator"],
                      benchmark=True,
                      callbacks=[ModelSummary(max_depth=1), checkpoint_callback, TQDMProgressBar()],
                      check_val_every_n_epoch=10,
                    #   val_check_interval=0.1,
                      default_root_dir=config['default_root_dir'],
                      max_epochs=config['num_epochs'], profiler="simple",
                      **optional_kw_args )

    Log.info("Start training...")

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=train_dataloader)