import os
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar

from utils.cpconfig import get_config
from train.dpinet_module import DPINet_Module
from train.invarnet_module import InvarNet_Module

from utils.train_utils import PhysionDynamicsDataModule, MOViDataModule


def main():
    pl_modules = {
        'dpinet': DPINet_Module,
        'invarnet': InvarNet_Module,
    }

    cfg = get_config()
    pl.seed_everything(62, workers=True)

    if cfg['data']['dataset_class'] != 'movi':
        datamodule = PhysionDynamicsDataModule(cfg)
    else:
        datamodule = MOViDataModule(cfg)
    model = pl_modules[cfg['model']['model_name']](cfg)

    # Setup Tensorboard logger
    logger = TensorBoardLogger(cfg['exp_path'], name='', version='', default_hp_metric=False, flush_secs=60)

    # Setup checkpoint saving and lr monitor callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg['exp_path'], every_n_train_steps=cfg['train']['save_after'], save_top_k=10, monitor='val/loss/total', save_last=True)
    lr_monitor_callback = LearningRateMonitor('step')
    pbar_callback = TQDMProgressBar(refresh_rate=10)
    callbacks = [checkpoint_callback, lr_monitor_callback, pbar_callback]

    # Setup PyTorch Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg['exp_path'],
        logger=logger,
        devices=1,
        accelerator='gpu',
        strategy=None,
        # plugins=DDPPlugin(find_unused_parameters=False),
        num_sanity_val_steps=2,
        log_every_n_steps=100 if 'log_after' not in cfg['train'] else cfg['train']['log_after'],
        callbacks=callbacks,
        max_epochs=cfg['train']['num_epochs'],
        check_val_every_n_epoch=10 if 'val_after' not in cfg['train'] else cfg['train']['val_after'],
        # gradient_clip_algorithm=None,
        # gradient_clip_val=None,
        accumulate_grad_batches=None if 'accu_grad_batches' not in cfg['train'] else cfg['train']['accu_grad_batches'],
        # enable_progress_bar=False,
        # limit_train_batches=1,
        # limit_val_batches=1,
        # profiler='advanced',
    )

    # Set checkpoint path if needed
    ckpt_path = cfg['model']['ckpt_path']
    if ckpt_path == 'resume_training': ckpt_path = os.path.join(cfg['exp_path'], 'last.ckpt')
    elif ckpt_path == '':              ckpt_path = None

    # If resuming the same experiment, set appropriate global step
    if cfg['model']['ckpt_path'] == 'resume_training' or cfg['model']['ckpt_path'].startswith(cfg['exp_path']):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        global_step_offset = checkpoint["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset
        del checkpoint

    if 'only_validate' in cfg['train'] and cfg['train']['only_validate']:
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        return

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()