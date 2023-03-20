import numpy as np
from dataset.particle_data_utils import find_relations_neighbor_scene, careful_revoxelized
from dataset.particle_data_utils import get_particle_poses_and_vels_2
from utils.viz_utils import dcn


def get_type_enc(num: int, etype: str, allowed_types: list):
    """Create one-hot encoding of shape (num, K) where K=len(allowed_types). Put 1 at the index of etype in allowed_types."""
    assert etype in allowed_types, "Node type must be one of {}".format(allowed_types)
    attr = np.zeros((num, len(allowed_types)))
    attr[:, allowed_types.index(etype)] = 1
    return attr

def get_type_idx(num: int, etype: str, allowed_types: list):
    """Create index vector a of shape (num,) where the array entries denote the index of etype in allowed_types."""
    assert etype in allowed_types, "Node type must be one of {}".format(allowed_types)
    attr = np.ones(num, dtype=int) * allowed_types.index(etype)
    return attr


class InvarNetGraph():
    """A handler class for graph creation.
    
    `prep_graph` is the main function that creates the graph.
    """
    def __init__(self, data_config, assets=None) -> None:
        self.assets = assets
        self.particle_type = data_config['particle_type']
        self.skip = data_config['skip']
        self.spacing = data_config['spacing'] # used for runtime particles
        self.radius = data_config['f1_radius'] * self.skip
        self.floor_dist_thresh = data_config['f1_radius'] * self.skip

        self.same_obj_rels = data_config['same_obj_rels']
        self.is_movi = 'movi' in data_config['dataset_class']

        self.node_allowed_types = ['rigid_0', 'rigid_1', 'root']  # allowing two materials
        self.rel_allowed_types = ['leaf-leaf', 'leaf-root', 'root-root', 'root-leaf']
        self.stage_allowed_types = ['leaf-leaf', 'leaf-root', 'root-root', 'root-leaf']

    def get_inter_object_rels(self, graph, seq_data):
        """Add relations across objects in the scene."""
        assert len(graph['rels']) == 0, "Rels need to be empty in inter-object rels function."
        poses = seq_data['cur_poses']
        instance_idx = dcn(seq_data['instance_idx'])

        # get rels
        inter_rels = find_relations_neighbor_scene(poses, instance_idx, self.radius, self.same_obj_rels)

        # get rel feats
        disp = poses[inter_rels[:, 0]] - poses[inter_rels[:, 1]]  # R x 3
        disp_norm = np.linalg.norm(disp, ord=2, axis=1, keepdims=True) # R x 1
        rel_type_enc = get_type_enc(inter_rels.shape[0], 'leaf-leaf', self.rel_allowed_types) # R x K
        inter_rel_feats = np.concatenate([disp, disp_norm, rel_type_enc],1)

        # get rel stages
        inter_rel_stages = get_type_idx(inter_rels.shape[0], 'leaf-leaf', self.stage_allowed_types) # R

        # add them to the graph
        graph['rels'] = np.concatenate([graph['rels'], inter_rels], 0)
        graph['rel_feats'] = np.concatenate([graph['rel_feats'], inter_rel_feats], 0)
        graph['stages'] = np.concatenate([graph['stages'], inter_rel_stages], 0)
        return graph


    def get_intra_object_rels(self, graph, seq_data):
        """Add relations within objects in the scene.
        Here we add a root node for each object and connect it to the leaves points.
        Helps to pass collision information instantly across the object.
        """

        poses, coms, com_vels = seq_data['cur_poses'], seq_data['cur_coms'], seq_data['cur_com_vels']
        instance_idx = dcn(seq_data['instance_idx'])
        intra_node_feats, intra_rels, intra_rel_feats, intra_stages = [], [], [], []
        num_nodes = graph['node_feats'].shape[0]

        root_poses = []
        for oi, (st, en) in enumerate(zip(instance_idx[:-1], instance_idx[1:])):
            # Add root node
            root_pos = coms[oi][None, :]
            root_vel = com_vels[oi][None, :]
            root_idx = num_nodes + oi

            root_enc = get_type_enc(1, 'root', self.node_allowed_types)  # 1 x NN
            root_dist_to_floor = np.clip(root_pos[:, 1:2], -self.floor_dist_thresh, self.floor_dist_thresh)  # 1 x 1
            root_feats = np.concatenate([root_vel, root_dist_to_floor, root_enc], 1)  # 1 x (3+1+NN)
            intra_node_feats.append(root_feats)
            root_poses.append(root_pos)

            disp = root_pos - poses[st:en]
            disp_norm = np.linalg.norm(disp, ord=2, axis=1, keepdims=True)

            # leaf-root
            rl = np.stack([np.ones(en-st, dtype=int)*root_idx, np.arange(st, en, dtype=int)], 1)
            rlenc = get_type_enc(rl.shape[0], 'leaf-root', self.rel_allowed_types)
            rlf = np.concatenate([disp, disp_norm, rlenc],1)
            intra_rels.append(rl)
            intra_rel_feats.append(rlf)
            intra_stages.append(get_type_idx(en-st, 'leaf-root', self.stage_allowed_types))

            # root-root
            rl = np.array([[root_idx, root_idx]])
            rlf = np.concatenate([np.zeros([1, 3]), np.zeros([1, 1]), get_type_enc(1, 'root-root', self.rel_allowed_types)],1)
            intra_rels.append(rl)
            intra_rel_feats.append(rlf)
            intra_stages.append(get_type_idx(1, 'root-root', self.stage_allowed_types))

            # root-leaf
            rl = np.stack([np.arange(st, en, dtype=int), np.ones(en-st, dtype=int)*root_idx], 1)
            rlenc = get_type_enc(rl.shape[0], 'root-leaf', self.rel_allowed_types)
            rlf = np.concatenate([-disp, disp_norm, rlenc],1)
            intra_rels.append(rl)
            intra_rel_feats.append(rlf)
            intra_stages.append(get_type_idx(en-st, 'root-leaf', self.stage_allowed_types))

        graph['root_poses'] = np.concatenate(root_poses, 0)
        graph['node_feats'] = np.concatenate([graph['node_feats']] + intra_node_feats, 0)
        graph['rels'] = np.concatenate([graph['rels']] + intra_rels, 0)
        graph['rel_feats'] = np.concatenate([graph['rel_feats']] + intra_rel_feats, 0)
        graph['stages'] = np.concatenate([graph['stages']] + intra_stages, 0)
        return graph


    def prep_graph(self, seq_data, noise_std=None):
        """Make graph nodes and relations for the current frame.
        
        This is the main function that prepares the graph.
        It first adds particles to the scene if not already done. It then find relations between particles.
        It also computes node features and relation features.
        
        Output is a dictionary with keys:
            node_feats: V x d1
            rels: R x 2
            rel_feats: R x d2
            stages: R. It denotes the order of processing for each relation.
        """

        seq_data = self.add_particles(seq_data)  # add particles if not already done
        seq_data = get_particle_poses_and_vels_2(seq_data, noise_std=noise_std)  # 3 x OV x 3

        # Node feature is [cur_vels, node_type_enc, dist_to_floor]

        # base particles
        cur_poses, cur_vels = seq_data['cur_poses'], seq_data['cur_vels']
        instance_idx = dcn(seq_data['instance_idx'])
        num_pts = instance_idx[1:] - instance_idx[:-1]
        node_type_enc = np.concatenate([ get_type_enc(npt, self.node_allowed_types[midx], self.node_allowed_types) for midx, npt in zip(dcn(seq_data['material']), num_pts)]) # OV x NN
        dist_to_floor = np.clip(cur_poses[:, 1:2], -self.floor_dist_thresh, self.floor_dist_thresh)  # OV x 1
        node_feats = np.concatenate([cur_vels, dist_to_floor, node_type_enc], 1)  # OV x (3+1+NN)

        # build rest of graph
        graph = {
            'node_feats': node_feats,  # OV x (3+NN+1)
            'rels': np.zeros([0, 2], dtype=int),  # no relations yet
            'rel_feats': np.zeros([0, 8]),
            'stages': np.zeros([0,]),
        }
        # Get inter-object relations
        graph = self.get_inter_object_rels(graph, seq_data)
        # Get intra-object relations
        graph = self.get_intra_object_rels(graph, seq_data)

        return graph

    def add_particles(self, seq_data):
        """Add particles to the scene if not already done."""
        assert self.particle_type in ['rt_same_spacing']

        if 'obj_points' in seq_data:
            return seq_data
        scale = seq_data['scale']
        instance_idx = [0]
        obj_points, verts, faces, coms = [], [], [], []
        for oi, shape_label in enumerate(seq_data['shape_label']):
            nms = self.assets[shape_label]['mesh'].copy().apply_transform(np.diag(scale[oi].tolist() + [1]))
            vrt = nms.vertices
            verts.append(vrt)
            faces.append(nms.faces)
            coms.append(nms.center_mass)

            if self.particle_type == 'rt_same_spacing':
                vox = self.assets[shape_label]['vox']
                # We get 3D particles by taking the centers of voxels. But we only have voxels of canonical objects.
                # vox.shape is the shape of 3D voxel grid. It is like a bounding box dimensions in voxel space.
                # vox.scale makes it the bbox of canonical object. scale[oi] then makes it the bbox of the object in current sequence.
                # self.spacing is the desired distance between two particles. We can revoxelize with the new shape to get the particles.
                new_shape = (np.array(vox.shape) * vox.scale * scale[oi] / self.spacing).astype(int)
                new_shape = np.maximum(new_shape, 1*np.ones_like(new_shape))
                pts = careful_revoxelized(vox, new_shape).points * scale[oi]
                # print(oi, shape_label, pts.shape, vox.shape, vox.scale, scale[oi], spacing, new_shape)
            else:
                raise AttributeError(f"{self.particle_type} not implemented")

            obj_points.append(pts)
            instance_idx.append(instance_idx[-1] + pts.shape[0])
        seq_data['obj_points'] = obj_points  # O x [V x 3]
        seq_data['instance_idx'] = np.array(instance_idx)
        seq_data['faces'] = faces
        seq_data['verts'] = verts
        seq_data['coms'] = np.stack(coms)
        return seq_data