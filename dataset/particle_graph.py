import pickle as pkl
import h5py
import numpy as np
import glob
from functools import partial
import os
import copy
from dataset.particle_data_utils import find_relations_neighbor_scene, remove_large_objects, subsample_particles_on_large_objects, careful_revoxelized
from dataset.particle_data_utils import get_imp_particles, get_surface_particles
from utils.viz_utils import dcn


def get_type_enc(num, etype, allowed_types):
    assert etype in allowed_types, "Node type must be one of {}".format(allowed_types)
    attr = np.zeros((num, len(allowed_types)))
    attr[:, allowed_types.index(etype)] = 1
    return attr

def get_type_idx(num, etype, allowed_types):
    assert etype in allowed_types, "Node type must be one of {}".format(allowed_types)
    attr = np.ones(num, dtype=int) * allowed_types.index(etype)
    return attr


def get_mesh_rels(poses, instance_idx, faces):
    mesh_rels = []
    for oi, (st, en) in enumerate(zip(instance_idx[:-1], instance_idx[1:])):
        assert faces[oi].max() < en - st, "Face indices must be within object range."
        mrels = faces[oi][:, [0, 1, 1, 0, 1, 2, 2, 1, 0, 2, 2, 0]].reshape(-1, 2) + st
        mesh_rels.append(mrels)
    mesh_rels = np.concatenate(mesh_rels, 0)
    return mesh_rels


class InvarNetGraph():
    def __init__(self, data_config, assets=None) -> None:
        self.assets = assets
        self.particle_type = data_config['particle_type']
        self.skip = data_config['skip']
        self.spacing = data_config['spacing'] # used for runtime particles
        self.radius = data_config['f1_radius'] * self.skip
        self.floor_dist_thresh = data_config['f1_radius'] * self.skip

        self.same_obj_rels = data_config['same_obj_rels'] if 'same_obj_rels' in data_config else False
        self.old_movi_setting = data_config['old_movi_setting']
        self.is_movi = 'movi' in data_config['dataset_class']
        self.add_density = data_config['add_density']

        if not self.is_movi or self.old_movi_setting:
            self.node_allowed_types = ['rigid', 'root']
        else:
            self.node_allowed_types = ['rigid_0', 'rigid_1', 'root']
        self.get_node_type_enc_fn = partial(get_type_enc, allowed_types=self.node_allowed_types)
        self.get_rel_type_enc_fn = partial(get_type_enc, allowed_types=['leaf-leaf', 'leaf-root', 'root-root', 'root-leaf'])
        self.get_stage_num_fn = partial(get_type_idx, allowed_types=['leaf-leaf', 'leaf-root', 'root-root', 'root-leaf'])

    def get_inter_object_rels(self, poses, instance_idx, graph, faces=None):
        # poses: OV x 3, instance_idx: O+1
        assert len(graph['rels']) == 0, "Rels need to be empty in inter-object rels function."

        # get rels
        inter_rels = find_relations_neighbor_scene(poses, instance_idx, self.radius, self.same_obj_rels)
        if self.particle_type == 'mesh_verts':
            inter_rels = np.concatenate([inter_rels, get_mesh_rels(poses, instance_idx, faces)], 0)

        # get rel feats
        disp = poses[inter_rels[:, 0]] - poses[inter_rels[:, 1]]
        disp_norm = np.linalg.norm(disp, ord=2, axis=1, keepdims=True)
        rel_type_enc = self.get_rel_type_enc_fn(inter_rels.shape[0], 'leaf-leaf')
        inter_rel_feats = np.concatenate([disp, disp_norm, rel_type_enc],1)

        # get rel stages
        inter_rel_stages = self.get_stage_num_fn(inter_rels.shape[0], 'leaf-leaf')

        graph['rels'] = np.concatenate([graph['rels'], inter_rels], 0)
        graph['rel_feats'] = np.concatenate([graph['rel_feats'], inter_rel_feats], 0)
        graph['stages'] = np.concatenate([graph['stages'], inter_rel_stages], 0)
        return graph


    def get_intra_object_rels(self, poses, vels, density, instance_idx, graph):
        # poses: OV x 3, vels: OV x 3, instance_idx: O+1

        intra_node_feats, intra_rels, intra_rel_feats, intra_stages = [], [], [], []
        num_nodes = graph['node_feats'].shape[0]

        root_poses = []
        for oi, (st, en) in enumerate(zip(instance_idx[:-1], instance_idx[1:])):
            # Add root node
            root_pos = poses[st:en].mean(0, keepdims=True)
            root_vel = vels[st:en].mean(0, keepdims=True)
            root_density = density[st:en].mean(0, keepdims=True)
            root_idx = num_nodes + oi

            root_enc = self.get_node_type_enc_fn(1, 'root')  # OV x NN
            root_dist_to_floor = np.clip(root_pos[:, 1:2], -self.floor_dist_thresh, self.floor_dist_thresh)  # OV x 1
            root_feats = np.concatenate([root_vel, root_dist_to_floor, root_density, root_enc], 1)  # OV x (3+1*+1+NN)
            intra_node_feats.append(root_feats)
            root_poses.append(root_pos)

            disp = root_pos - poses[st:en]
            disp_norm = np.linalg.norm(disp, ord=2, axis=1, keepdims=True)

            # leaf-root
            rl = np.stack([np.ones(en-st, dtype=int)*root_idx, np.arange(st, en, dtype=int)], 1)
            rlenc = self.get_rel_type_enc_fn(rl.shape[0], 'leaf-root')
            rlf = np.concatenate([disp, disp_norm, rlenc],1)
            intra_rels.append(rl)
            intra_rel_feats.append(rlf)
            intra_stages.append(self.get_stage_num_fn(en-st, 'leaf-root'))

            # root-root
            rl = np.array([[root_idx, root_idx]])
            rlf = np.concatenate([np.zeros([1, 3]), np.zeros([1, 1]), self.get_rel_type_enc_fn(1, 'root-root')],1)
            intra_rels.append(rl)
            intra_rel_feats.append(rlf)
            intra_stages.append(self.get_stage_num_fn(1, 'root-root'))

            # root-leaf
            rl = np.stack([np.arange(st, en, dtype=int), np.ones(en-st, dtype=int)*root_idx], 1)
            rlenc = self.get_rel_type_enc_fn(rl.shape[0], 'root-leaf')
            rlf = np.concatenate([-disp, disp_norm, rlenc],1)
            intra_rels.append(rl)
            intra_rel_feats.append(rlf)
            intra_stages.append(self.get_stage_num_fn(en-st, 'root-leaf'))

        graph['root_poses'] = np.concatenate(root_poses, 0)
        graph['node_feats'] = np.concatenate([graph['node_feats']] + intra_node_feats, 0)
        graph['rels'] = np.concatenate([graph['rels']] + intra_rels, 0)
        graph['rel_feats'] = np.concatenate([graph['rel_feats']] + intra_rel_feats, 0)
        graph['stages'] = np.concatenate([graph['stages']] + intra_stages, 0)
        return graph


    def prep_graph(self, cur_poses, cur_vels, seq_data):
        """Make graph nodes and relations for the current frame."""
        # Node feature is [cur_vels, offset_to_com, node_type_enc, dist_to_floor]
        instance_idx = dcn(seq_data['instance_idx'])

        # basic particles
        if not self.is_movi or self.old_movi_setting:
            node_type_enc = self.get_node_type_enc_fn(cur_poses.shape[0], 'rigid')  # OV x NN
        else:
            mat = [f'rigid_{x}' for x in dcn(seq_data['material'])]
            num_pts = instance_idx[1:] - instance_idx[:-1]
            node_type_enc = np.concatenate([self.get_node_type_enc_fn(npt, mat) for mat, npt in zip(mat, num_pts)])
        dist_to_floor = np.clip(cur_poses[:, 1:2], -self.floor_dist_thresh, self.floor_dist_thresh)  # OV x 1
        density = dcn(seq_data['density'])[:, None] if self.add_density else np.zeros([cur_vels.shape[0], 0])
        node_feats = np.concatenate([cur_vels, dist_to_floor, density, node_type_enc], 1)  # OV x (3+1*+1+NN)

        # build rest of graph
        graph = {
            'node_feats': node_feats,  # OV x (3+3+NN+1)
            'rels': np.zeros([0, 2], dtype=int),
            'rel_feats': np.zeros([0, 8]),
            'stages': self.get_stage_num_fn(0, 'leaf-leaf'),
        }
        # Get inter-object relations
        graph = self.get_inter_object_rels(cur_poses, instance_idx, graph, faces=[dcn(f) for f in seq_data['faces']])
        # Get intra-object relations
        graph = self.get_intra_object_rels(cur_poses, cur_vels, density, instance_idx, graph)

        if self.is_movi and self.old_movi_setting:
            graph['node_feats'][:instance_idx[-1], -2] = np.repeat(dcn(seq_data['material']), instance_idx[1:] - instance_idx[:-1])
        return graph


    def rt_variable_sampling(self, seq_data, cur_transform):
        if 'base_obj_points' not in seq_data:
            old_particle_type = self.particle_type
            self.particle_type = 'rt_same_spacing'
            seq_data = self.add_particles(seq_data)
            self.particle_type = old_particle_type
            seq_data['base_obj_points'], seq_data['base_instance_idx'] = seq_data['obj_points'].copy(), seq_data['instance_idx'].copy()

        if self.particle_type == 'rt_imp_sampling':
            var_pts = get_imp_particles(seq_data['shape_label'], self.radius, self.assets, cur_transform[0], seq_data['scale'])
        elif self.particle_type == 'rt_surface_sampling':
            # import IPython; IPython.embed()
            var_pts = get_surface_particles(seq_data['shape_label'], self.radius, self.assets, cur_transform[0], seq_data['scale'])
        else: raise NotImplementedError

        new_obj_points, new_instance_idx = copy.deepcopy(seq_data['base_obj_points']), [0]
        for oi in range(len(var_pts)):
            new_obj_points[oi] = np.concatenate([new_obj_points[oi], var_pts[oi]])
            new_instance_idx.append(new_instance_idx[-1] + new_obj_points[oi].shape[0])
        seq_data['obj_points'] = new_obj_points
        seq_data['instance_idx'] = np.array(new_instance_idx)
        seq_data['n_particles'] = new_instance_idx[-1]
        return seq_data

    def add_particles(self, seq_data, cur_transform=None):
        assert self.particle_type in ['fish', 'mesh_verts', 'rt_same_spacing', 'rt_same_obj_verts', 'rt_imp_sampling',
                                      'rt_surface_sampling']

        if self.particle_type == 'fish':
            # particles already there. just need to clean it.
            seq_data = remove_large_objects(seq_data)
            seq_data = subsample_particles_on_large_objects(seq_data)
            del seq_data['instance']  # it's a byte array, not tensor-able and not needed now
            return seq_data

        if self.particle_type in ['rt_imp_sampling', 'rt_surface_sampling']:
            return self.rt_variable_sampling(seq_data, cur_transform)

        scale = seq_data['scale']
        instance_idx = [0]
        obj_points, verts, faces, density = [], [], [], []
        for oi, shape_label in enumerate(seq_data['shape_label']):
            nms = self.assets[shape_label]['mesh'].copy().apply_transform(np.diag(scale[oi].tolist() + [1]))
            if self.particle_type == 'mesh_verts':
                nms = nms.subdivide_to_size(self.spacing*3)
            vrt = nms.vertices
            verts.append(vrt)
            faces.append(nms.faces)
            if self.particle_type == 'mesh_verts':
                pts = vrt
            elif self.particle_type == 'rt_same_spacing':
                vox = self.assets[shape_label]['vox']
                new_shape = (np.array(vox.shape) * vox.scale * scale[oi] / self.spacing).astype(int)
                new_shape = np.maximum(new_shape, 1*np.ones_like(new_shape))
                pts = careful_revoxelized(vox, new_shape).points * scale[oi]
                # print(oi, shape_label, pts.shape, vox.shape, vox.scale, scale[oi], spacing, new_shape)
            elif self.particle_type == 'rt_same_obj_verts':
                vox = self.assets[shape_label]['vox']
                new_shape = (np.array(vox.shape) * vox.scale * scale[oi])
                new_shape = new_shape * (1536 * np.prod(vox.shape) / np.prod(new_shape) / vox.filled_count) ** (1./3)
                new_shape = new_shape.astype(int)
                new_shape = np.maximum(new_shape, 1*np.ones_like(new_shape))
                # pts = careful_revoxelized(vox, new_shape).points * scale[oi]
                new_vox = careful_revoxelized(vox, new_shape)
                pts = new_vox.points * scale[oi]
                density.append(np.ones(pts.shape[0]) * (new_vox.scale * scale[oi] / self.spacing).mean())
                # print(oi, shape_label, pts.shape, vox.shape, vox.scale, scale[oi], spacing, new_shape)
            obj_points.append(pts)
            instance_idx.append(instance_idx[-1] + pts.shape[0])
        seq_data['obj_points'] = obj_points  # O x [V x 3]
        seq_data['instance_idx'] = np.array(instance_idx)
        seq_data['n_particles'] = instance_idx[-1]
        if density:
            seq_data['density'] = np.concatenate(density, 0)
        seq_data['faces'] = faces
        seq_data['verts'] = verts
        return seq_data