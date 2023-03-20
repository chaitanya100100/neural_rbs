import os
import torch
import numpy as np
import pytorch_lightning as pl
import datetime

from dataset.data_utils import recursive_to_tensor
from utils.cpconfig import get_config
from train.invarnet_module import InvarNet_Module
from utils.train_utils import MOViDataModule
from utils.train_utils import recursive_to
from utils.viz_utils import save_video, append_to_file


def main():
    cfg = get_config()
    device = torch.device('cuda:0')
    pl.seed_everything(62, workers=True)

    pl_modules = {
        # 'dpinet': DPINet_Module,
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
        raise ArithmeticError
    else:
        datamodule = MOViDataModule(cfg)
        datamodule.setup(eval_split)
        dataset = datamodule.val_dataset if eval_split == 'val' else datamodule.train_dataset
        if viz:
            seq_ids = np.unique(np.linspace(0, len(dataset.all_hdf5) - 1, 100, dtype=int)).tolist()
        else:
            seq_ids = list(range(len(dataset.all_hdf5)))

    ckpt_path = cfg['model']['ckpt_path']
    if ckpt_path == 'resume_training' or ckpt_path == '':
        ckpt_path = os.path.join(cfg['exp_path'], 'last.ckpt')
    print("Loading checkpoint path: ", ckpt_path)
    model = pl_modules[cfg['model']['model_name']].load_from_checkpoint(ckpt_path, cfg=cfg).eval().to(device)

    angle_err, trans_err = [], []
    for i, seq_id in enumerate(seq_ids):
        seq_data = dataset.get_seq_data(seq_id, 'all')
        seq_data['graph'] = dataset.graph_builder.prep_graph(seq_data)
        seq_data['graph_builder'] = dataset.graph_builder
        seq_data = recursive_to(recursive_to_tensor(seq_data), device)
        
        ang_err_seq, trans_err_seq, imgs = model.validation_step_rollout(seq_data, batch_idx=None, ret_viz=viz)

        if viz:
            save_video(imgs, os.path.join(out_dir, '{}.webm'.format(seq_data['seq_name'])), fps=int(1./seq_data['dt'].item()))

        # transforms error
        angle_err.append(ang_err_seq)
        trans_err.append(trans_err_seq)

        cur_metric = "Angle Err: {:.3f}, Trans Err: {:.3f}".format(np.mean(angle_err), np.mean(trans_err))
        toprint = "{} {} {} {}".format(i, len(seq_ids), cur_metric, seq_data['seq_name'])
        print(toprint)
        append_to_file(out_file, toprint + "\n")


if __name__ == '__main__':
    main()