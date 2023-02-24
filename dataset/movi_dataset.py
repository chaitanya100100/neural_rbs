import argparse
import os
import numpy as np
import trimesh
from tqdm import tqdm
import xml.etree.ElementTree as ET
import pickle as pkl
from torch.utils.data import Dataset
from scipy.spatial.transform.rotation import Rotation
import h5py
from functools import partial
import cv2

from dataset.particle_data_utils import get_particle_poses_and_vels, add_noise, careful_revoxelized, transform_points, get_transformation_matrix
from dataset.data_utils import store_hdf5_data, nested_shallow_dict, process_movi_tfds_example, recursive_to_tensor

from dataset.particle_graph import InvarNetGraph


KUBASIC_OBJECTS = ("cube", "cylinder", "sphere", "cone", "torus", "gear",
                   "torus_knot", "sponge", "spot", "teapot", "suzanne")
KUBASIC_MASSES = [0.9936, 0.7808, 0.5225, 0.4774, 0.2627, 0.2842, 0.2724, 0.5476, 0.2763, 0.3227, 0.3113]
ZUP_TO_YUP = [
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
]


def calculate_base_masses(asset_dir):
    def get_mass(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        assert root[0][0][1].tag == 'mass'
        return float(root[0][0][1].attrib['value'])
    masses = []
    for o in KUBASIC_OBJECTS:
        xml_file = os.path.join(asset_dir, o, 'object.urdf')
        mass = get_mass(xml_file)
        masses.append(mass)
    print(masses)


def save_kubric_assets():
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':10'
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset_dir', type=str, required=True)
    parser.add_argument('--vox_dim', type=int, required=True)
    args = parser.parse_args()
    for asset_id in KUBASIC_OBJECTS:
        print(asset_id)
        for obj_name in ['collision_geometry.obj', 'visual_geometry.obj']:
            mesh_path = os.path.join(args.asset_dir, asset_id + '/' + obj_name)
            obj_mesh = trimesh.load_mesh(mesh_path)
            vox_path = mesh_path.rsplit(".")[0] + ".binvox"
            vox_mesh_path = mesh_path.rsplit(".")[0] + "_vox_mesh.obj"
            surface_path = mesh_path.rsplit(".")[0] + "_surface_points.obj"

            # fine_mesh = obj_mesh.subdivide_to_size(0.1)
            # fine_mesh = obj_mesh.subdivide_loop(1)
            # fine_mesh = obj_mesh.simplify_quadratic_decimation(obj_mesh.faces.shape[0])

            pts, _ = trimesh.sample.sample_surface_even(obj_mesh, 500)
            # pts = np.concatenate([pts, obj_mesh.vertices.view(np.ndarray)], 0)
            with open(surface_path, 'w') as f:
                f.write(trimesh.exchange.export.export_obj(trimesh.Trimesh(vertices=pts)))
            continue

            kwargs = {'use_offscreen_pbuffer': False, 'dilated_carving': True, 'wireframe':True, 'dimension': args.vox_dim}
            vox = obj_mesh.voxelized(pitch=None, method='binvox', **kwargs)
            with open(vox_path, 'wb') as f:
                f.write(trimesh.exchange.binvox.export_binvox(vox))

            with open(vox_mesh_path, 'w') as f:
                f.write(trimesh.exchange.export.export_obj(careful_revoxelized(vox, [s//4 for s in vox.shape]).as_boxes()))


def download_movi_dataset():
    import tensorflow_datasets as tfds
    download_dir = "/ccn2/u/chpatel/kubric_data"
    dataset_name = "movi_b"
    data_root = os.path.join(download_dir, dataset_name)
    os.makedirs(data_root, exist_ok=False)
    ds, ds_info = tfds.load(dataset_name, data_dir="gs://kubric-public/tfds", with_info=True)
    for mode in ['train', 'validation']:
        os.makedirs(os.path.join(data_root, mode), exist_ok=False)
        for idx, example in enumerate(tqdm(tfds.as_numpy(ds[mode]))):
            example = nested_shallow_dict(example)
            example = process_movi_tfds_example(example)
            store_hdf5_data(example.keys(), example.values(), os.path.join(data_root, mode, str(idx) + '.hdf5'))


def get_assets(asset_dir):
    obj_meshes = {}
    for shape_label, shape_name in enumerate(KUBASIC_OBJECTS):
        ms = trimesh.load_mesh(os.path.join(asset_dir, shape_name, 'collision_geometry.obj'))
        if isinstance(ms.visual, trimesh.visual.texture.TextureVisuals):
            ms.visual = ms.visual.to_color()
        ms.visual.vertex_colors[:, :3] = 0

        with open(os.path.join(asset_dir, shape_name, 'collision_geometry.binvox'), 'rb') as fpt:
            vox = trimesh.exchange.binvox.load_binvox(fpt)
        surface = trimesh.load_mesh(os.path.join(asset_dir, shape_name, 'visual_geometry_surface_points.obj'))

        obj_meshes[shape_label] = {'mesh': ms, 'vox': vox, 'surface': surface}
    return obj_meshes


def get_scale(shape_label, material_label, mass):
    base_masses = np.array([KUBASIC_MASSES[sid] for sid in shape_label])
    mults = np.zeros_like(base_masses)
    mults[material_label == 0] = 2.7
    mults[material_label == 1] = 1.1
    scale = np.cbrt(mass / base_masses / mults)
    return scale


def load_seq_data(data_path, skip_frame=None, randomize_skipping=False):
    assert skip_frame is None or skip_frame == 1, "MOVi shouldn't need skipping"
    with h5py.File(data_path, 'r') as h5:
        dct = {
            'mass': h5['instances/mass'][()],  # O
            'material': h5['instances/material_label'][()],  # O
            'trans': h5['instances/positions'][()],  # O x N x 3
            'rot': h5['instances/quaternions'][()],  # O x N x 4
            'shape_label': h5['instances/shape_label'][()], # O
            'dt': 1./12,
            'imgs': h5['video'][()],  # N x H x W x 3,
            'cam_pos': h5['camera/positions'][()][0],  # assuming camera is static
        }
        dct['rot'] = dct['rot'][:, :, [1,2,3,0]]  # nasty nasty nasty thing that can be a silent bug
        dct['rot'] = np.transpose(dct['rot'], [1,0,2])  # N x O x 4
        dct['trans'] = np.transpose(dct['trans'], [1,0,2])  # N x O x 3
    
        # zup to yup
        qzy = Rotation.from_matrix(np.array(ZUP_TO_YUP)[:3, :3])
        new_rot = qzy * Rotation.from_quat(dct['rot'].reshape((-1, 4)))
        dct['rot'] = new_rot.as_quat().reshape(dct['rot'].shape)
        dct['trans'] = (np.array(ZUP_TO_YUP)[:3, :3] @ dct['trans'].reshape([-1, 3]).T).T.reshape(dct['trans'].shape)
        dct['cam_pos'] = np.array(ZUP_TO_YUP)[:3,:3] @ dct['cam_pos']
    
        # add scale properly
        if 'scale' in h5['instances']:
            scale = h5['instances/scale'][()]
        elif 'size_label' in h5['instances']:
            size_label = h5['instances/size_label'][()]
            scale = np.where(size_label.astype(bool), np.ones(size_label.shape[0]) * 1.4, np.ones(size_label.shape[0]) * 0.7)
        else:
            scale = get_scale(h5['instances/shape_label'][()], h5['instances/material_label'][()], h5['instances/mass'][()])
        dct['scale'] = np.tile(scale[:, None], [1, 3])
        dct['transform'] = get_transformation_matrix(dct['rot'], dct['trans'])
    
    dct.update({
        'time_step': dct['rot'].shape[0],
        'filename': data_path,
        'seq_name': os.path.basename(data_path).replace('.hdf5', ''),
        'start_frame': 0, 'yellow_id': 0, 'red_id': 0, 'label': 0,
    })
    return dct


class MOViDataset(Dataset):
    def __init__(self, data_config):
        super().__init__()

        self.skip = data_config['skip']
        self.subset = data_config['subset']
        assert self.subset in ['movi_a', 'movi_b']
        self.split = data_config['split']
        assert self.split in ['train', 'val']
        self.noise = 3.e-4 if data_config['split'] == 'train' else 0
        print("Adding noise: ", self.noise)
        self.assets = get_assets(os.path.join(data_config['data_dir'], 'assets'))

        data_dir = os.path.join(data_config['data_dir'], self.subset, {'train': 'train', 'val': 'validation'}[self.split])
        self.all_hdf5 = sorted([os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.endswith('.hdf5')])

        self.graph_type = data_config['graph_type']
        self.particle_type = data_config['particle_type']
    
        self.spacing = data_config['spacing'] # used for runtime particles
        self.avg_frames_per_seq = 24 // self.skip
        print("Particle type: {} | Graph type: {} | Spacing: {:.4f}".format(self.particle_type, self.graph_type, self.spacing))

        self.val_rollout = True #and self.particle_type != 'rt_imp_sampling'
        self.graph_builder = InvarNetGraph(data_config, self.assets)

    def __len__(self):
        if self.split != 'train' and self.val_rollout:
            return len(self.all_hdf5)
        return len(self.all_hdf5) * self.avg_frames_per_seq

    def get_seq_data(self, seq_idx, fr_idxs):
        """Get sequence static data and dynamic data for the given frame indices."""

        assert fr_idxs in ['all', 'random3']
        filename = self.all_hdf5[seq_idx]
        seq_data = load_seq_data(filename, skip_frame=self.skip, randomize_skipping=(fr_idxs == 'random3'))
        if fr_idxs == 'random3':
            fidx = np.random.randint(seq_data['start_frame'], seq_data['time_step']-3)  # exclude initial frames with external stimuli
            stf, enf, focf = fidx, fidx+3, fidx+1
        elif fr_idxs == 'all':
            stf, enf, focf = None, None, 1
        seq_data['cur_transform'] = [seq_data['transform'][focf], seq_data['transform'][focf-1]]

        seq_data = self.graph_builder.add_particles(seq_data, cur_transform=seq_data['cur_transform'])
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
        seq_data = recursive_to_tensor(seq_data)
        return seq_data


if __name__ == '__main__':
    # download_movi_dataset()
    save_kubric_assets()
    raise AttributeError

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
        dataset_class: movi
        data_dir: /ccn2/u/chpatel/kubric_data/
        subset: movi_a
        skip: 1
        split: 'train'
        spacing: 0.2
        f1_radius: 0.6
        add_density: False
        graph_type: invarnet
        particle_type: rt_surface_sampling
        old_movi_setting: False
    """)
    data_config = cfg['data']
    dataset = MOViDataset(cfg['data'])

    out_dir = '/ccn2/u/rmvenkat/chpatel/test/videos/movi_check'
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
        elif dataset.particle_type in ['rt_imp_sampling', 'rt_surface_sampling']:
            imgs = []
            for i in range(poses.shape[0]):
                seq_data['cur_transform'] = [seq_data['transform'][i], seq_data['transform'][i-1]] if i!=0 else [seq_data['transform'][i]]*2
                seq_data = dataset.graph_builder.add_particles(seq_data, seq_data['cur_transform'])
                cur_nodes = np.concatenate(transform_points(seq_data['obj_points'], seq_data['cur_transform'][0], None))
                im = viz_points(dcn(cur_nodes), seq_data['instance_idx'], cam_pos=seq_data['cam_pos'], add_axis=True, add_floor=True)
                imgs.append(im)
            imgs = np.stack(imgs)
        else:
            imgs = []
            for i in range(poses.shape[0]):
                graph = dataset.graph_builder.prep_graph(poses[i], vels[i], seq_data)
                cur_nodes = dcn(poses[i])
                cur_nodes = np.concatenate([dcn(poses[i]), dcn(graph['root_poses'])], 0)
                # im = viz_points(dcn(cur_nodes), seq_data['instance_idx'], graph['rels'], cam_pos=seq_data['cam_pos'], add_axis=True, add_floor=True)
                im = viz_points(dcn(cur_nodes), seq_data['instance_idx'], cam_pos=seq_data['cam_pos'], add_axis=True, add_floor=True)
                imgs.append(im)
            imgs = np.stack(imgs)

        rgb = np.stack([cv2.resize(x, imgs.shape[-3:-1]) for x in dcn(seq_data['imgs'])])
        save_video(np.concatenate([rgb, imgs], -2), f'{out_dir}/{sid}.webm', fps=int(1./seq_data['dt']))

    import ipdb; ipdb.set_trace()
