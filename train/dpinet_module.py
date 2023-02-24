import os
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from model.dynamic.dpinet import DPINet
from utils.viz_utils import viz_points, dcn, put_text_on_img
from utils.train_utils import recursive_to
from dataset.data_utils import recursive_to_tensor


class DPINet_Module(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.train_config = cfg['train'] if 'train' in cfg else None
        self.model_config = cfg['model']

        # Model
        self.model = DPINet(self.model_config)

    def configure_optimizers(self):
        wd = float(self.train_config['weight_decay']) if 'weight_decay' in self.train_config else 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.train_config['lr']), weight_decay=wd, amsgrad=False)
        self.optimizer = optimizer
        if not self.train_config['lr_scheduler']:
            return optimizer
        assert self.train_config['lr_scheduler'] == 'plateau'
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)
        return [optimizer], [{
            'scheduler': scheduler,
            'monitor': 'val/loss/total',
            'interval': 'epoch',
        }]

    def on_train_start(self):
        if self.model_config['ckpt_path'] == '':
            return
        print()
        print("Changing learning rate of loaded checkpoint.")
        # import IPython; IPython.embed()
        for g in self.optimizer.param_groups:
            g['lr'] = self.train_config['lr']

    def forward(self, batch):
        graph = batch['graph']
        pred_vels = self.model(graph['nodes'], graph['node_attrs'], graph['rels'], graph['rel_attrs'], graph['stages'], graph['prop_steps'], batch['instance_idx'], batch['dt'])
        pred_poses = batch['cur_poses'] + pred_vels * batch['dt']
        return {'pred_vels': pred_vels, 'pred_poses': pred_poses}

    def compute_losses(self, batch, out, log_prefix=None):
        # losses
        losses = {}
        vels_loss = (out['pred_vels'] - batch['target_vels']).square().mean(-1)

        # put same weight on each object
        npts = batch['instance_idx'][1:] - batch['instance_idx'][:-1]
        weight = torch.ones_like(npts).div(npts*npts.shape[0]/npts.sum()).repeat_interleave(npts)
        losses['loss/vels'] = (vels_loss * weight).mean()

        # Final loss and logging
        losses['loss/total'] = sum(losses.values())
        num_pts = batch['cur_poses'].shape[0]
        if log_prefix:
            for k, v in losses.items():
                self.log(f'{log_prefix}/{k}', v.item(), on_epoch=True, batch_size=num_pts)

        self.log(f'{log_prefix}/zero_pred_loss', batch['target_vels'].square().mean().item(), on_epoch=True, batch_size=num_pts)
        self.log(f'{log_prefix}/vels_error', (out['pred_vels'] - batch['target_vels']).detach().square().sum(-1).sqrt().mean().item(), on_epoch=True, batch_size=num_pts)
        # if torch.any(batch['target_vels'].abs() > 1.e-3).item():
        #     self.logger.experiment.add_histogram(f'{log_prefix}/target_vel', batch['target_vels'][batch['target_vels'].abs() > 1.e-3], self.global_step, bins=100)
        # self.logger.experiment.add_histogram(f'{log_prefix}/target_pos', batch['target_poses'], self.global_step, bins=100)
        return losses

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        losses = self.compute_losses(batch, out, log_prefix='train')
        if self.global_step % self.cfg['train']['viz_after'] == 0:
            self.visualize(batch, out, 'train')
        return losses['loss/total']

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        losses = self.compute_losses(batch, out, log_prefix='val')
        if batch_idx % self.cfg['train']['viz_after'] == 0:
            self.visualize(batch, out, 'val')
        return losses['loss/total']

    def visualize(self, batch, out, prefix):
        graph = batch['graph']
        cur_poses = dcn(batch['cur_poses'])
        target_poses = dcn(batch['target_poses'])
        instance_idx = dcn(batch['instance_idx'])

        cur_nodes = dcn(graph['nodes'][:, :3])
        cur_rels = dcn(graph['rels'])
        cur_rels = cur_rels[ dcn(graph['stages']) == 0 ]
        pred_poses = dcn(out['pred_poses'])

        cur_img = viz_points(cur_nodes, instance_idx, lines=cur_rels)
        # cur_img = viz_points(cur_poses, instance_idx)
        # cur_img = put_text_on_img(cur_img.copy(), 'input')
        target_img = viz_points(target_poses, instance_idx)
        # target_img = put_text_on_img(target_img.copy(), 'target')
        pred_img = viz_points(pred_poses, instance_idx)
        # pred_img = put_text_on_img(pred_img.copy(), 'pred')

        img = np.concatenate([cur_img, target_img, pred_img], axis=1)
        if prefix is not None:
            self.logger.experiment.add_image(f'{prefix}/pred', img, self.global_step,  dataformats='HWC')
        return img


    def rollout(self, cur_poses, cur_vels, instance_idx, dt, num_steps, prep_graph_fn, ret_rels=False):
        device = cur_poses.device
        seq_poses, seq_vels = [cur_poses], [cur_vels]
        rels = []

        for _ in range(num_steps-1):
            graph = prep_graph_fn(dcn(seq_poses[-1]), dcn(seq_vels[-1]), dcn(instance_idx))
            if ret_rels:
                rels.append(graph['rels'])
            batch = {
                'graph': recursive_to(recursive_to_tensor(graph), device),
                'instance_idx': instance_idx, 'cur_poses': seq_poses[-1], 'dt': dt}

            with torch.no_grad():
                out = self.forward(batch)
            
            seq_poses.append(out['pred_poses'])
            seq_vels.append(out['pred_vels'])

        seq_poses = torch.stack(seq_poses)
        seq_vels = torch.stack(seq_vels)

        if ret_rels:
            graph = prep_graph_fn(dcn(seq_poses[-1]), dcn(seq_vels[-1]), dcn(instance_idx))
            rels.append(graph['rels'])
            return seq_poses, seq_vels, rels
        return seq_poses, seq_vels


def check_ocp(poses, instance_idx, id1, id2, dist_thresh):
    # poses: N x OV x 3
    poses_ob1 = poses[:, instance_idx[id1]:instance_idx[id1+1]]
    poses_ob2 = poses[:, instance_idx[id2]:instance_idx[id2+1]]

    ocp = False
    for pt1, pt2 in zip(poses_ob1, poses_ob2):
        sim_mat = scipy.spatial.distance_matrix(pt1, pt2, p=2)
        ocp = ocp or (np.min(sim_mat) < dist_thresh)
        if ocp:
            break
    return ocp

def rollout_error(gt_poses, pred_poses):
    min_len = min(gt_poses.shape[0], pred_poses.shape[0])
    gt_poses = gt_poses[:min_len]
    pred_poses = pred_poses[:min_len]
    err = np.square(gt_poses - pred_poses)
    err = np.sqrt(np.sum(err, -1))
    return np.mean(err)