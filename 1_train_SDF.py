import time
import torch

from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, ModelSummary

from utils.logging import Log
from utils import parse_config_from_args

from pathlib import Path
from model.SDFAutoEncoder import SDFAutoEncoder
from model.SDFAutoEncoder.dataloader import GenSDFDataset

if __name__ == '__main__':
    wandb_logger = WandbLogger(log_model="all")
    torch.set_float32_matmul_precision('high')

    run_name = time.strftime("%m/%d %I%p:%M:%S")
    config = parse_config_from_args()
    seed_everything(20030912)

    model = SDFAutoEncoder(config)

    # Configure data module
    d_configs = config['dataset_n_dataloader']
    dataloader = [
               DataLoader(GenSDFDataset(dataset_dir=Path(d_configs['dataset_dir']), train=is_train,
                                        samples_per_mesh=d_configs['samples_per_mesh'],
                                        pc_size=d_configs['pc_size'],
                                        uniform_sample_ratio=d_configs['uniform_sample_ratio']),
                num_workers=d_configs['n_workers'], batch_size=d_configs['batch_size'],
                drop_last=True, shuffle=is_train, pin_memory=True, persistent_workers=True)
            for is_train in [True, False]
    ]


    # Configure save checkpoint callback
    checkpoint_callback = ModelCheckpoint(
            save_top_k=10,
            mode="min",
            monitor="loss",
            save_last=True,
            every_n_train_steps=config['checkpoint']['freq'],
            dirpath=config['checkpoint']['path'] + '/' + run_name,
            filename="{epoch:04d}-{loss:.5f}",
        )

    # Configure trainer
    optional_kw_args = dict()
    optional_kw_args['logger'] = wandb_logger

    trainer = Trainer(devices=config['devices'], accelerator=config["accelerator"],
                      benchmark=True,
                      callbacks=[ModelSummary(max_depth=1), checkpoint_callback, TQDMProgressBar()],
                      check_val_every_n_epoch=config['evaluation']['freq'],
                      default_root_dir=config['default_root_dir'],
                      max_epochs=config['num_epochs'], profiler="simple",
                      **optional_kw_args )

    Log.info("Start training...")

    trainer.fit(model=model, train_dataloaders=dataloader[0],
                               val_dataloaders=dataloader[1])