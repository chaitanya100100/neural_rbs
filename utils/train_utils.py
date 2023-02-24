import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import glob
import pytorch_lightning as pl
from tensorboard.backend.event_processing import event_accumulator
import json
import copy
import scipy
from scipy.spatial.transform.rotation import Rotation
import warnings

from dataset.movi_dataset import MOViDataset
from dataset.physion_particle_dataset import PhysionParticleDataset
from dataset.data_utils import get_dynamics_seq_paths, get_readout_seq_paths, collate_fn, SCENARIOS


class PhysionDynamicsDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def get_dynamics_dataset(data_config, split=None):

        assert split in [None, 'train', 'val']
        split_percentage = data_config['split_percentage'] if 'split_percentage' in data_config else None
        if split is not None: assert split_percentage is not None

        seq_paths = get_dynamics_seq_paths(data_config['physion_path'], data_config['protocol'], split, split_percentage)
        data_config = copy.deepcopy(data_config)
        data_config['split'] = split
        ds = PhysionParticleDataset(seq_paths, data_config)
        print(data_config['protocol'], split, "Dataset size", len(ds))
        return ds

    def setup(self, stage):
        cfg = self.cfg

        self.train_dataset = self.get_dynamics_dataset(cfg['data'], split='train')
        self.val_dataset = self.get_dynamics_dataset(cfg['data'], split='val')

        self.batch_size = cfg['train']['batch_size']
        self.num_workers = 8
        torch.multiprocessing.set_sharing_strategy('file_system')
        import warnings; warnings.filterwarnings("ignore", ".*does not have many workers which may be a bottleneck. Consider increasing.*")

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers, drop_last=True, shuffle=True, pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers, drop_last=False, shuffle=False, pin_memory=True)
        return val_dataloader


class MOViDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def get_movi_dataset(data_config, split=None):
        dcfg = copy.deepcopy(data_config)
        dcfg['split'] = split
        ds = MOViDataset(dcfg)
        print(split, "Dataset size", len(ds))
        return ds

    def setup(self, stage):
        cfg = self.cfg
        assert cfg['data']['dataset_class'] == 'movi'

        self.train_dataset = self.get_movi_dataset(cfg['data'], split='train')
        self.val_dataset = self.get_movi_dataset(cfg['data'], split='val')

        self.batch_size = cfg['train']['batch_size']
        self.num_workers = 8
        torch.multiprocessing.set_sharing_strategy('file_system')
        import warnings; warnings.filterwarnings("ignore", ".*does not have many workers which may be a bottleneck. Consider increasing.*")

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers, drop_last=True, shuffle=True, pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers, drop_last=False, shuffle=False, pin_memory=True)
        return val_dataloader


def recursive_to(x, target: torch.device):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x


def get_angle_error(r1, r2):
    # r1, r2: O x 4 or N x O x 4 or N x 3 x 3 or N x O x 3 x 3
    if r1.shape[-1] == 4:
        r1, r2 = r1.reshape([-1,4]), r2.reshape([-1,4])
        diff = Rotation.from_quat(r2).inv() * Rotation.from_quat(r1)
    elif r1.shape[-1] == 3 and r1.shape[-2] == 3:
        r1, r2 = r1.reshape([-1,3,3]), r2.reshape([-1,3,3])
        diff = Rotation.from_matrix(r2).inv() * Rotation.from_matrix(r1)
    else:
        raise ValueError("Invalid shape for rotation")
    return diff.magnitude().mean()

def get_trans_error(t1, t2):
    t1, t2 = t1.reshape([-1,3]), t2.reshape([-1,3])
    d = (t1 - t2) ** 2
    d = np.sqrt(d.sum(-1))
    return d.mean()


class StatsMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.sq_sum = 0
        self.std = 0
        self.min = None
        self.max = None

    def update_batch(self, val):

        if isinstance(val, torch.Tensor):
            if self.min is None: self.min = torch.tensor(1.e9)
            if self.max is None: self.max = torch.tensor(-1.e9)
        else:  # np.ndarray
            if self.min is None: self.min = np.array(1.e9)
            if self.max is None: self.max = np.array(-1.e9)  
        
        self.min = min(self.min, min(val))
        self.max = max(self.max, max(val))

        self.sum += sum(val)
        self.count += len(val)
        self.avg = self.sum / self.count
        self.sq_sum += sum(val**2)
        self.std = (self.sq_sum / self.count - self.avg ** 2) ** 0.5

    def update(self, val, n=1):
        if not isinstance(val, (int, float)):
            assert len(val.shape) == 1
            self.update_batch(val)
            return

        self.val = val
        if isinstance(val, int) or isinstance(val, float):
            min_fn, max_fn = min, max
            if self.min is None: self.min = 1.e9
            if self.max is None: self.max = -1.e9
        elif isinstance(val, torch.Tensor):
            min_fn, max_fn = torch.minimum, torch.maximum
            if self.min is None: self.min = torch.tensor(1.e9)
            if self.max is None: self.max = torch.tensor(-1.e9)
        else:  # np.ndarray
            min_fn, max_fn = np.minimum, np.maximum
            if self.min is None: self.min = np.array(1.e9)
            if self.max is None: self.max = np.array(-1.e9)
        self.min = min_fn(self.min, val)
        self.max = max_fn(self.max, val)

        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += (val**2) * n
        self.std = (self.sq_sum / self.count - self.avg ** 2) ** 0.5