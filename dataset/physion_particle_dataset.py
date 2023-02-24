import pickle as pkl
import h5py
import numpy as np
import glob
import os
from scipy.spatial.transform.rotation import Rotation
from torch.utils.data import Dataset
import trimesh
import cv2
from functools import partial

from dataset.data_utils import recursive_to_tensor, recursive_to_numpy, ReadoutLabels, get_dynamics_seq_paths, index_img
from dataset.particle_data_utils import get_particle_poses_and_vels, add_noise, seq_datapath_to_fish_datapath, remove_carpet
from dataset.particle_graph import InvarNetGraph


PHYSION_BASIC_ASSETS = ('cube', 'platonic', 'triangular_prism', 'cone', 'cylinder', 'bowl', 'pentagon', 
                        'pipe', 'pyramid', 'torus', 'octahedron', 'sphere')


def save_physion_assets():
    seq_paths = get_dynamics_seq_paths('/ccn2/u/chpatel/physion_dataset/', 'only_Dominoes')
    seq_paths += get_dynamics_seq_paths('/ccn2/u/chpatel/physion_dataset/', 'only_Drop')
    asset_dir = '/ccn2/u/chpatel/physion_dataset/assets/'
    vox_dim = 128

    if 'PYOPENGL_PLATFORM' not in os.environ:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':10'
    assets = []
    for spath in seq_paths:
        with h5py.File(spath) as h5:
            model_names = h5['static']['model_names'][()].tolist()
            ignores = h5['static']['distractors'][()].tolist() + h5['static']['occluders'][()].tolist()
            model_names = [m.decode('utf-8') for m in model_names]
            ignores = [m.decode('utf-8') for m in ignores]
            for oi, mname in enumerate(model_names):
                if mname in ignores or mname in assets:
                    continue
                print(mname, ignores)
                faces = h5['static']['mesh'][f'faces_{oi}'][()]
                verts = h5['static']['mesh'][f'vertices_{oi}'][()]
                obj_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                # obj_mesh = trimesh.Trimesh(vertices=obj_mesh.vertices + 0.05 * obj_mesh.vertex_normals, faces=obj_mesh.faces)
                with open(os.path.join(asset_dir, '{}.obj'.format(mname)), 'w') as fpt:
                    fpt.write(trimesh.exchange.obj.export_obj(obj_mesh))

                spacing = obj_mesh.extents / vox_dim
                new_bounds = obj_mesh.bounds.copy()
                new_bounds[0] -= 0.48*spacing
                new_bounds[1] += 0.48*spacing
                kwargs = {'use_offscreen_pbuffer': False, 'dilated_carving': True, 'wireframe':True, 'dimension': vox_dim, 
                # 'bounds': new_bounds
                }
                vox = obj_mesh.voxelized(pitch=None, method='binvox', **kwargs)
                with open(os.path.join(asset_dir, '{}.binvox'.format(mname)), 'wb') as f:
                    f.write(trimesh.exchange.binvox.export_binvox(vox))
                assets.append(mname)
    print(assets)


def get_assets(asset_dir):
    obj_meshes = {}
    for oi, mname in enumerate(PHYSION_BASIC_ASSETS):
        ms = trimesh.load_mesh(os.path.join(asset_dir, mname + '.obj'))
        ms.visual.vertex_colors[:, :3] = 0
        with open(os.path.join(asset_dir, mname + '.binvox'), 'rb') as fpt:
            vox = trimesh.exchange.binvox.load_binvox(fpt)
        obj_meshes[mname] = {'mesh': ms, 'vox': vox,}
    return obj_meshes


def load_seq_data_fish(data_path, skip_frame=None, randomize_skipping=False):
    with open(data_path + '/phases_dict.pkl', "rb") as fp:
        dct = pkl.load(fp)
    with h5py.File(data_path + '/dynamic_data.h5', 'r') as h5:
        trans = h5['obj_positions'][:]
        rot = h5['obj_rotations'][:]
    time_step = dct['time_step']
    dt = dct['dt']
    if skip_frame is not None:
        if not randomize_skipping: idxs = list(range(0, time_step, skip_frame))
        else: idxs = list(range(np.random.randint(skip_frame), time_step, skip_frame))
        rot = rot[idxs]
        trans = trans[idxs]
        time_step = len(idxs)
        dt *= skip_frame
    ret = {
        'trans': trans,
        'rot': rot,
        'dt': dt,
        'yellow_id': dct['yellow_id'], 'red_id': dct['red_id'], 'time_step': time_step,
        'seq_name': os.path.basename(data_path).replace('.hdf5', ''),
        'cam_pos': np.array([0., 1.25, 3.]),
        'instance_idx': np.array(dct['instance_idx']),
        'obj_points': dct['obj_points'],
        'instance': dct['instance'], 'clusters': dct['clusters'],
        'start_frame': (30+2)//skip_frame+1,
    }
    return ret


def load_seq_data(data_path, skip_frame=None, randomize_skipping=False):
    skip_frame = 1 if skip_frame is None else skip_frame
    with h5py.File(data_path) as h5:
        # static data
        model_names = h5['static/model_names'][()].tolist()
        ignores = h5['static/distractors'][()].tolist() + h5['static/occluders'][()].tolist()
        model_names = [m.decode('utf-8') for m in model_names]
        ignores = [m.decode('utf-8') for m in ignores]
        keep_obj = [(mname not in ignores) for mname in model_names]

        model_names = [mname for mname, keep in zip(model_names, keep_obj) if keep]
        dt = 0.01 * skip_frame
        scale = h5['static/scale'][()][keep_obj]
        mass = h5['static/mass'][()]
        num_frames = len(h5['frames'].keys())
        # camT = h5['frames/0000/camera_matrices/camera_matrix_cam0'][()].reshape([4,4])
        # cam_pos = -(camT[:3,:3].T @ camT[:3,-1:])[:, 0]
        cam_pos = np.array([0., 1.25, 2.5])
        start_time = (h5['static/push_time'][()]+2) // skip_frame + 1
        
        # get red and yellow id
        object_ids = h5['static/object_ids'][()].tolist()
        yellow_id = object_ids.index(h5["static/zone_id"][()])
        red_id = object_ids.index(h5["static/target_id"][()])
        yellow_id = np.where(keep_obj)[0].tolist().index(yellow_id)
        red_id = np.where(keep_obj)[0].tolist().index(red_id)

        # dynamic data
        if not randomize_skipping: fids = list(range(0, num_frames, skip_frame))
        else: fids = list(range(np.random.randint(skip_frame), num_frames, skip_frame))

        imgs, trans, rot = [], [], []
        for fi in fids:
            x = '{:04d}'.format(fi)
            imgs.append(index_img(h5, fi)[0])

            suffix = '' if 'positions' in h5[f'frames/{x}/objects'].keys() else '_cam0'
            trans.append(h5[f'frames/{x}/objects/positions'+suffix][()][keep_obj])
            rot.append(h5[f'frames/{x}/objects/rotations'+suffix][()][keep_obj])

    ret = {
        'imgs': np.stack(imgs),
        'trans': np.stack(trans),
        'rot': np.stack(rot),
        'scale': scale, 'mass': mass, 'dt': dt, 'fids': fids,
        'yellow_id': yellow_id, 'red_id': red_id, 'time_step': len(fids),
        'seq_name': os.path.basename(data_path).replace('.hdf5', ''),
        'shape_label': model_names,
        'cam_pos': cam_pos,
        'start_frame': start_time,
    }
    return ret


class PhysionParticleDataset(Dataset):
    def __init__(self, hdf5_list, data_config):
        super().__init__()
        self.all_hdf5 = hdf5_list
        self.skip = data_config['skip']
        self.split = data_config['split']
        self.noise = 3.e-4 if data_config['split'] == 'train' else 0
        print("Adding noise: ", self.noise)
        self.readout_labels = ReadoutLabels(os.path.join(data_config['physion_path'], 'ocp'))
        self.assets = get_assets(os.path.join(data_config['physion_path'], 'assets'))
        self.remove_carpet = data_config['remove_carpet'] if 'remove_carpet' in data_config else False
    
        self.graph_type = data_config['graph_type']
        self.particle_type = data_config['particle_type']
    
        self.spacing = data_config['spacing'] # used for runtime particles
        self.avg_frames_per_seq = 150 // self.skip
        print("Particle type: {} | Graph type: {} | Spacing: {:.4f}".format(self.particle_type, self.graph_type, self.spacing))

        self.val_rollout = True

        self.graph_builder = InvarNetGraph(data_config)

    def __len__(self):
        if self.split != 'train' and self.val_rollout:
            return len(self.all_hdf5)
        return len(self.all_hdf5) * self.avg_frames_per_seq

    def get_seq_data(self, seq_idx, fr_idxs):
        """Get sequence static data and dynamic data for the given frame indices."""

        assert fr_idxs in ['all', 'random3']
        filename = self.all_hdf5[seq_idx]
        if self.particle_type == 'fish':
            seq_data = load_seq_data_fish(seq_datapath_to_fish_datapath(filename), skip_frame=self.skip, randomize_skipping=(fr_idxs == 'random3'))
        else:
            seq_data = load_seq_data(filename, skip_frame=self.skip, randomize_skipping=(fr_idxs == 'random3'))
        seq_data['label'] = self.readout_labels.get(filename)
        if self.remove_carpet: seq_data = remove_carpet(seq_data)

        if fr_idxs == 'random3':
            fidx = np.random.randint(seq_data['start_frame'], seq_data['time_step']-3)  # exclude initial frames with external stimuli
            stf, enf, focf = fidx, fidx+3, fidx+1
        elif fr_idxs == 'all':
            stf, enf, focf = None, None, 0

        seq_data = self.graph_builder.add_particles(seq_data, self.spacing, self.graph_builder.radius, self.particle_type, self.assets, focus_frame=focf)
        poses, vels = get_particle_poses_and_vels(seq_data, stf, enf)  # 3 x OV x 3

        return seq_data, poses, vels

    def __getitem__(self, idx):
        if self.split != 'train' and self.val_rollout:
            seq_data, poses, vels = self.get_seq_data(idx, 'all')
            ret = {'seq_data': seq_data, 'poses': poses, 'vels': vels, 'graph_builder': self.graph_builder}
            return recursive_to_tensor(ret)

        seq_data, poses, vels = self.get_seq_data(idx % len(self.all_hdf5), 'random3')
        if self.noise > 0:
            poses, vels = add_noise(poses, vels, self.noise, seq_data['dt'], 'cumu')

        cur_poses, cur_vels = poses[1], vels[1]  # OV x 3
        nxt_poses, nxt_vels = poses[2], vels[2]  # OV x 3

        graph = self.graph_builder.prep_graph(cur_poses, cur_vels, seq_data)

        seq_data.update({
            'target_poses': nxt_poses,
            'target_vels': nxt_vels,
            'cur_poses': cur_poses,
            'cur_vels': cur_vels,
            'graph': graph,
        })
        # for k, v in seq_data.items():
        #     print(k, type(v))
        seq_data = recursive_to_tensor(seq_data)
        return seq_data


if __name__ == '__main__':
    # save_physion_assets()
    # raise AttributeError
    import os
    import yaml
    from dataset.data_utils import get_dynamics_seq_paths
    from utils.viz_utils import viz_points, save_video, dcn
    from utils.renderer import MeshViewer
    import pytorch_lightning as pl
    pl.seed_everything(10)

    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['DISPLAY'] = ':0.0'
    cfg = yaml.safe_load("""
    data:
        dataset_class: physion_particle
        physion_path: /ccn2/u/chpatel/physion_v2_hlink/
        particle_type: mesh_verts
        spacing: 0.033
        f1_radius: 0.05
        skip: 3
        remove_carpet: False
        graph_type: invarnet
        protocol: 'only_Drop'  # allbut_scenario, only_scenario, all
        split_percentage: 0.9
        split: train
    """)
    seq_paths = get_dynamics_seq_paths( cfg['data']['physion_path'],  cfg['data']['protocol'])
    dataset = PhysionParticleDataset(seq_paths, cfg['data'])
    out_dir = '/ccn2/u/rmvenkat/chpatel/test/videos/pts_v2_meshverts_orig'
    os.makedirs(out_dir, exist_ok=True)

    for sid in np.random.randint(0, len(dataset.all_hdf5), size=(10,)):
        seq_data, poses, vels = dataset.get_seq_data(sid, 'all')

        if dataset.particle_type == 'mesh_verts' and False:
            imgs = []
            instance_idx = seq_data['instance_idx']
            mv = MeshViewer(use_offscreen=True, width=512, height=512, cam_pos=seq_data['cam_pos'], add_axis=True, add_floor=False)
            for oi, (st, en) in enumerate(zip(instance_idx[:-1], instance_idx[1:])):
                mv.add_mesh_seq([trimesh.Trimesh(vertices=poses[i, st:en], 
                                                faces=seq_data['faces'][oi]) for i in range(poses.shape[0])])
            imgs = mv.animate()
            del mv
        else:
            imgs = []
            for i in range(poses.shape[0]):
                graph = dataset.graph_builder.prep_graph(poses[i], vels[i], seq_data)
                cur_nodes = dcn(poses[i])
                cur_nodes = np.concatenate([dcn(poses[i]), dcn(graph['root_poses'])], 0)
                im = viz_points(dcn(cur_nodes), seq_data['instance_idx'], graph['rels'], cam_pos=seq_data['cam_pos'], add_axis=True, add_floor=False)
                # im = viz_points(dcn(cur_nodes), seq_data['instance_idx'], cam_pos=seq_data['cam_pos'], add_axis=True, add_floor=False)
                imgs.append(im)
            imgs = np.stack(imgs)

        rgb = np.stack([cv2.resize(x, imgs.shape[-3:-1]) for x in dcn(seq_data['imgs'])])
        save_video(np.concatenate([rgb, imgs], -2), f'{out_dir}/{sid}.webm', fps=int(1./seq_data['dt']))

    import ipdb; ipdb.set_trace()