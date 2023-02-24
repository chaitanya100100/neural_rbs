import os
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy
import cv2
import copy

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from model.dynamic.invarnet import InvarNet
from utils.viz_utils import viz_points, dcn, put_text_on_img
from utils.train_utils import recursive_to, get_angle_error, get_trans_error
from dataset.data_utils import recursive_to_tensor, recursive_to_numpy
from dataset.particle_data_utils import shape_matching_objects, transform_points


class InvarNet_Module(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.train_config = cfg['train'] if 'train' in cfg else None
        self.model_config = cfg['model']

        # Model
        self.model = InvarNet(self.model_config)
        if self.model_config['node_feat_dim'] == 6:
            self.register_buffer('nf_mean', torch.tensor([0, 0, 0, 0.1, 0, 0]).float()[None], persistent=False)
            self.register_buffer('nf_std', torch.tensor([0.15, 0.15, 0.15, 0.07, 1, 1]).float()[None], persistent=False)
        else:
            self.register_buffer('nf_mean', torch.tensor([0, 0, 0, 0.1, 0, 0, 0]).float()[None], persistent=False)
            self.register_buffer('nf_std', torch.tensor([0.15, 0.15, 0.15, 0.07, 1, 1, 1]).float()[None], persistent=False)
        self.register_buffer('rf_mean', torch.tensor([0, 0, 0, .1, 0, 0, 0, 0]).float()[None], persistent=False)
        self.register_buffer('rf_std', torch.tensor([0.07, 0.07, 0.07, 0.07, 1, 1, 1, 1]).float()[None], persistent=False)

    def configure_optimizers(self):
        wd = float(self.train_config['weight_decay']) if 'weight_decay' in self.train_config else 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.train_config['lr']), weight_decay=wd, amsgrad=False)
        self.optimizer = optimizer
        if not self.train_config['lr_scheduler']:
            return optimizer
        assert self.train_config['lr_scheduler'] == 'plateau'
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=15, verbose=True)
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
        node_feats = (graph['node_feats'] - self.nf_mean) / self.nf_std
        rel_feats = (graph['rel_feats'] - self.rf_mean) / self.rf_std
        out = self.model.forward(node_feats, graph['rels'], rel_feats, graph['stages'])
        pred_vels = out[:batch['instance_idx'][-1]]
        pred_poses = batch['cur_poses'] + pred_vels * batch['dt'].item()
        ret = {'pred_vels': pred_vels, 'pred_poses': pred_poses}
        if not self.training:
            pred_poses_refined, transform = shape_matching_objects(np.concatenate([dcn(o) for o in batch['obj_points']]), dcn(pred_poses), dcn(batch['instance_idx']))
            ret.update({'transform': transform, 'pred_poses_refined': torch.from_numpy(pred_poses_refined).to(node_feats.device)})
        return ret

    def get_equal_obj_weight_pts(self, instance_idx):
        npts = instance_idx[1:] - instance_idx[:-1]
        weight = torch.ones_like(npts).div(npts*npts.shape[0]/npts.sum()).repeat_interleave(npts)
        return weight

    def compute_losses(self, batch, out, log_prefix=None):
        # losses
        losses = {}
        vels_loss = (out['pred_vels'] - batch['target_vels']).square().mean(-1)
        losses['loss/vels'] = (vels_loss * self.get_equal_obj_weight_pts(batch['instance_idx'])).mean()

        # Final loss and logging
        losses['loss/total'] = sum(losses.values())
        num_pts = batch['cur_poses'].shape[0]
        if log_prefix:
            for k, v in losses.items():
                self.log(f'{log_prefix}/{k}', v.item(), on_epoch=True, batch_size=num_pts)

        self.log(f'{log_prefix}/zero_pred_loss', batch['target_vels'].square().mean().item(), on_epoch=True, batch_size=num_pts)
        self.log(f'{log_prefix}/vels_error', (out['pred_vels'] - batch['target_vels']).detach().square().sum(-1).sqrt().mean().item(), on_epoch=True, batch_size=num_pts)
        # self.logger.experiment.add_histogram(f'{log_prefix}/target_pos', batch['target_poses'], self.global_step, bins=100)
        return losses

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        losses = self.compute_losses(batch, out, log_prefix='train')
        if self.global_step % self.cfg['train']['viz_after'] == 0:
            self.visualize(batch, out, 'train')
        return losses['loss/total']

    def validation_step_rollout(self, batch, batch_idx, log_prefix=None):
        seq_data, poses, vels, graph_builder = batch['seq_data'], batch['poses'], batch['vels'], batch['graph_builder']
        start_frame = seq_data['start_frame']
        instance_idx = seq_data['instance_idx']
        num_frames, _, _ = poses.shape
        pred_poses, _, seq_transforms, instance_idx = self.rollout(poses[start_frame], vels[start_frame], recursive_to_numpy(seq_data), num_frames-start_frame, graph_builder)

        ang_err = get_angle_error(dcn(seq_data['transform'][:, :3, :3]), seq_transforms[:num_frames, :3, :3])
        self.log(f'{log_prefix}/angle_error', ang_err, on_epoch=True, batch_size=num_frames)
        trans_err = get_trans_error(dcn(seq_data['transform'][:, :3, 3]), seq_transforms[:num_frames, :3, 3])
        self.log(f'{log_prefix}/trans_error', trans_err, on_epoch=True, batch_size=num_frames)

        losses = {}
        # pos_loss = (pred_poses[1:] - poses[start_frame+1:]).square().mean(-1)  # N x V
        # wt = self.get_equal_obj_weight_pts(instance_idx)  # V
        # # wt = torch.ones_like(wt)
        # losses['loss/rollout_poses'] = (pos_loss * wt[None, :]).mean()
        # losses['loss/total'] = sum(losses.values())
        losses['loss/total'] = ang_err + trans_err

        if log_prefix:
            for k, v in losses.items():
                self.log(f'{log_prefix}/{k}', v.item(), on_epoch=True, batch_size=num_frames)

            if batch_idx == 0:
                poses = dcn(poses)
                pred_poses = [dcn(pp) for pp in pred_poses] if isinstance(pred_poses, list) else dcn(pred_poses)
                cam_pos = dcn(seq_data['cam_pos'])
                gt_imgs = viz_points(poses[start_frame:], seq_data['instance_idx'], cam_pos=cam_pos, add_axis=True, img_size=256)
                pred_imgs = viz_points(pred_poses, instance_idx, cam_pos=cam_pos, add_axis=True, img_size=256)
                imgs = [gt_imgs, pred_imgs]
                if 'imgs' in seq_data:
                    rgb = np.stack([cv2.resize(x, gt_imgs.shape[-3:-1]) for x in dcn(seq_data['imgs'][start_frame:])], 0)
                    imgs.insert(0, rgb)
                imgs = np.concatenate(imgs, -2)
                self.logger.experiment.add_video(f'{log_prefix}/rollout', imgs[None].transpose([0, 1, 4, 2, 3]), self.global_step, fps=int(1./seq_data['dt']))

    def validation_step(self, batch, batch_idx):
        if 'seq_data' in batch:
            return self.validation_step_rollout(batch, batch_idx, log_prefix='val')

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

        cur_root_poses = dcn(graph['root_poses'])
        cur_nodes = np.concatenate([cur_poses, cur_root_poses], 0)
        cur_rels = dcn(graph['rels'])
        # cur_rels = cur_rels[ dcn(graph['stages']) == 0 ]
        pred_poses = dcn(out['pred_poses'])

        kwargs = {'add_axis': True}
        if 'cam_pos' in batch: kwargs['cam_pos'] = dcn(batch['cam_pos'])
        cur_img = viz_points(cur_nodes, instance_idx, lines=cur_rels, **kwargs)
        target_img = viz_points(target_poses, instance_idx, **kwargs)
        pred_img = viz_points(pred_poses, instance_idx, **kwargs)

        img = np.concatenate([cur_img, target_img, pred_img], axis=1)
        if prefix is not None:
            self.logger.experiment.add_image(f'{prefix}/pred', img, self.global_step,  dataformats='HWC')
        return img


    def rollout(self, cur_poses, cur_vels, seq_data, num_steps, graph_builder):
        device = cur_poses.device
        is_var_sampling = graph_builder.particle_type in ['rt_imp_sampling', 'rt_surface_sampling']
        if is_var_sampling:
            seq_data = copy.deepcopy(seq_data)
        dt = seq_data['dt']
        seq_poses = [cur_poses]
        seq_vels = [cur_vels]
        seq_transforms = [dcn(seq_data['cur_transform'][0])]
        seq_instance_idx = [seq_data['instance_idx']]

        for fi in range(num_steps-1):

            graph = graph_builder.prep_graph(dcn(seq_poses[-1]), dcn(seq_vels[-1]), seq_data)
            seq_data.update({'graph': recursive_to(recursive_to_tensor(graph), device), 'cur_poses': seq_poses[-1]})

            with torch.no_grad():
                out = self.forward(seq_data)
            
            # print(out['pred_poses'].shape, out['pred_poses_refined'].shape)
            seq_poses.append(out['pred_poses_refined'])
            seq_vels.append(out['pred_vels'])
            seq_transforms.append(out['transform'])

            if is_var_sampling:
                seq_data = graph_builder.rt_variable_sampling(seq_data, [seq_transforms[-2], seq_transforms[-1]])
                new_cur_poses = np.concatenate(transform_points(seq_data['obj_points'], seq_transforms[-1]))
                new_prev_poses = np.concatenate(transform_points(seq_data['obj_points'], seq_transforms[-2]))
                new_cur_vel = (new_cur_poses - new_prev_poses) / dt
                seq_poses[-1] = torch.from_numpy(new_cur_poses).to(device)
                seq_vels[-1] = torch.from_numpy(new_cur_vel).to(device)
                seq_instance_idx.append(seq_data['instance_idx'])

        if not is_var_sampling:
            seq_poses = torch.stack(seq_poses)
            seq_vels = torch.stack(seq_vels)

        seq_transforms = np.stack(seq_transforms)
        # if ret_rels:
        #     graph = graph_builder.prep_graph(dcn(seq_poses[-1]), dcn(seq_vels[-1]), seq_data)
        #     rels.append(graph['rels'])
        #     return seq_poses, seq_vels, rels
        if not is_var_sampling: seq_instance_idx = seq_instance_idx[0]
        return seq_poses, seq_vels, seq_transforms, seq_instance_idx
    
