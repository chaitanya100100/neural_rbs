import os
import torch
import numpy as np
import pytorch_lightning as pl
import cv2
import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from dataset.data_utils import recursive_to_tensor, MAX_ROLLOUT_LENGTH, READOUT_THRESHOLD
from utils.cpconfig import get_config
from train.dpinet_module import DPINet_Module, check_ocp, rollout_error
from train.invarnet_module import InvarNet_Module
from utils.train_utils import PhysionDynamicsDataModule, MOViDataModule
from utils.train_utils import recursive_to, get_angle_error, get_trans_error
from utils.viz_utils import save_video, dcn, print_dict, viz_points, append_to_file


def main():
    cfg = get_config()
    device = torch.device('cuda:0')
    pl.seed_everything(62, workers=True)

    pl_modules = {
        'dpinet': DPINet_Module,
        'invarnet': InvarNet_Module,
    }

    viz = cfg['eval']['viz']
    eval_split = cfg['eval']['split']
    out_file = os.path.join(cfg['exp_path'], 'eval_{}.txt'.format(eval_split))
    append_to_file(out_file, "\n\n\n{}\n".format(datetime.datetime.now()))
    if viz:
        out_dir = os.path.join(cfg['exp_path'], cfg['eval']['outdir_prefix'] + '_' + eval_split)
        os.makedirs(out_dir, exist_ok=True)

    if cfg['data']['dataset_class'] != 'movi':
        scenario = cfg['data']['protocol'].split('_')[1]
        if eval_split in ['train', 'val']:
            datamodule = PhysionDynamicsDataModule(cfg)
            datamodule.setup(eval_split)
            dataset = datamodule.val_dataset if eval_split == 'val' else datamodule.train_dataset
            seq_ids = np.unique(np.linspace(0, len(dataset.all_hdf5) - 1, 100, dtype=int)).tolist()
        elif eval_split == 'test':
            cfg['data']['scenario'] = scenario
            datamodule = ReadoutDataModule(cfg)
            datamodule.setup('test')
            dataset = datamodule.test_dataset
            seq_ids = range(0, len(dataset.all_hdf5))
        else:
            raise AttributeError("Unknown eval_split: {}".format(eval_split))

        thresh = READOUT_THRESHOLD[scenario] * 0.05    
    else:
        datamodule = MOViDataModule(cfg)
        datamodule.setup(eval_split)
        dataset = datamodule.val_dataset if eval_split == 'val' else datamodule.train_dataset
        seq_ids = np.unique(np.linspace(0, len(dataset.all_hdf5) - 1, 100, dtype=int)).tolist()
        thresh = 0.05

    ckpt_path = cfg['model']['ckpt_path']
    if ckpt_path == 'resume_training' or ckpt_path == '':
        ckpt_path = os.path.join(cfg['exp_path'], 'last.ckpt')
    print("Loading checkpoint path: ", ckpt_path)
    model = pl_modules[cfg['model']['model_name']].load_from_checkpoint(ckpt_path, cfg=cfg).eval().to(device)

    labels, gt_det_labels, pred_det_labels, angle_err, trans_err = [], [], [], [], []
    for i, seq_id in enumerate(seq_ids):
        seq_data, seq_poses, seq_vels = dataset.get_seq_data(seq_id, 'all')
        dt = seq_data['dt']
        start_frame = seq_data['start_frame']
        num_steps = int(seq_poses.shape[0] * 1.3) - start_frame
        cam_pos = seq_data['cam_pos']

        # predict from start_frame
        poses_inp, vels_inp = recursive_to(recursive_to_tensor([seq_poses[start_frame], seq_vels[start_frame] ]), device)
        pred_poses, _, pred_transforms, instance_idx = model.rollout(poses_inp, vels_inp, seq_data, num_steps, dataset.graph_builder)
        pred_poses = [dcn(pp) for pp in pred_poses] if isinstance(pred_poses, list) else dcn(pred_poses)

        # append [0, start_frame) to pred_poses
        if start_frame > 0:
            pred_poses = np.concatenate([
                np.stack([seq_poses[0]] + [seq_poses[fi-1] + seq_vels[fi] * dt for fi in range(1, start_frame)]),
                pred_poses], 0)
        assert len(pred_poses) == num_steps + start_frame

        if viz:
            gt_imgs = viz_points(seq_poses, seq_data['instance_idx'], cam_pos=cam_pos, add_axis=True, add_floor=True)
            pred_imgs = viz_points(pred_poses, instance_idx, cam_pos=cam_pos, add_axis=True, add_floor=True)

            def pad_it(arr, before, after):
                return np.pad(arr, pad_width=[(before, after), (0,0), (0,0), (0,0)], mode='edge')
            max_len = max(gt_imgs.shape[0], pred_imgs.shape[0])
            gt_imgs = pad_it(gt_imgs, 0, max_len-gt_imgs.shape[0])
            pred_imgs = pad_it(pred_imgs, 0, max_len-pred_imgs.shape[0])
            imgs = [gt_imgs, pred_imgs]
            if 'imgs' in seq_data:
                rgb = np.stack([cv2.resize(x, gt_imgs.shape[-3:-1]) for x in dcn(seq_data['imgs'])])
                rgb = pad_it(rgb, 0, max_len-rgb.shape[0])
                imgs.insert(0, rgb)
            save_video(np.concatenate(imgs, -2), os.path.join(out_dir, '{}.webm'.format(seq_data['seq_name'])), fps=int(1./dt))

        # gt_dl = check_ocp(seq_poses, instance_idx, seq_data['red_id'], seq_data['yellow_id'], dist_thresh=thresh)
        # pred_dl = check_ocp(pred_poses, instance_idx, seq_data['red_id'], seq_data['yellow_id'], dist_thresh=thresh)
        # rerr = rollout_error(seq_poses, pred_poses)
        gt_dl, pred_dl = 0, 0

        labels.append(int(seq_data['label']))
        gt_det_labels.append(int(gt_dl))
        pred_det_labels.append(int(pred_dl))
        # traj_error.append(rerr)
        cur_acc = np.mean(np.array(labels) == np.array(pred_det_labels))
        gt_acc = np.mean(np.array(labels) == np.array(gt_det_labels))
        cur_conf_mat = confusion_matrix(labels, pred_det_labels)
        gt_conf_mat = confusion_matrix(labels, gt_det_labels)

        # transforms error
        num_frames = seq_poses.shape[0]
        angle_err.append(get_angle_error(seq_data['transform'][:, :3, :3], pred_transforms[:num_frames, :3, :3]))
        trans_err.append(get_trans_error(seq_data['transform'][:, :3, 3], pred_transforms[:num_frames, :3, 3]) )

        cur_metric = "GT Acc: {:.3f}, GT Conf Mat: {}, Cur Acc: {:.3f}, Cur Conf Mat: {}, Angle Err: {:.3f}, Trans Err: {:.3f}".format(
            gt_acc, gt_conf_mat.reshape(-1), cur_acc, cur_conf_mat.reshape(-1), np.mean(angle_err), np.mean(trans_err))
        toprint = "{} {} {} {}".format(i, len(seq_ids), cur_metric, seq_data['seq_name'])
        print(toprint)
        append_to_file(out_file, toprint + "\n")


if __name__ == '__main__':
    main()