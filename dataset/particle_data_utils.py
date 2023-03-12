import numpy as np
from scipy.spatial.transform.rotation import Rotation
import scipy.spatial as spatial
import trimesh
import os
import warnings
import copy
import scipy


def get_colors():
    from utils.renderer import colors as colors_dict
    names = ['yellow', 'red'] + [c for c in colors_dict.keys() if c not in ['yellow', 'red']]
    return [colors_dict[n] for n in names]


def remove_carpet(dct):
    ridx = 0
    num_obj = dct['rot'].shape[1]
    keep_idx = [i!=ridx for i in range(num_obj)]
    dct['rot'] = dct['rot'][:, keep_idx]
    dct['trans'] = dct['trans'][:, keep_idx]
    dct['red_id'] = dct['yellow_id'] = 0
    if 'obj_points' in dct:
        dct['obj_points'] = [dct['obj_points'][i] for i in range(num_obj) if i!=ridx]
        assert 'instance_idx' in dct
        new_instance_idx = [0]
        for x in dct['obj_points']:
            new_instance_idx.append(new_instance_idx[-1] + len(x))
        dct['instance_idx'] = new_instance_idx
    if 'scale' in dct:
        dct['scale'] = dct['scale'][keep_idx]
    if 'shape_label' in dct:
        dct['shape_label'] = [dct['shape_label'][i] for i in range(num_obj) if i!=ridx]
    if 'mass' in dct:
        dct['mass'] = dct['mass'][keep_idx]
    if 'instance' in dct:
        dct['instance'] = dct['instance'][keep_idx]
    if 'clusters' in dct:
        dct['clusters'] = dct['clusters'][keep_idx]
    return dct


def remove_large_objects(dct):
    if dct["instance_idx"][-1] <= 3000:
        return dct

    # Find ids of objects NOT to remove
    critical_objects = [b'cloth_square', b'buddah', b'bowl', b'cone', b'cube', b'cylinder', b'dumbbell', b'octahedron', b'pentagon', b'pipe', b'platonic', b'pyramid', b'sphere', b'torus', b'triangular_prism']
    ok_ids = []
    for i, (st, en) in enumerate(zip(dct['instance_idx'][:-1], dct['instance_idx'][1:])):
        if en <= st:
            print("Warning: empty object")
            import ipdb; ipdb.set_trace()
        if dct['instance'][i] not in critical_objects and (en-st) > 3000:
            continue
        ok_ids.append(i)
    # ok_ids = [1]
    if len(ok_ids) == len(dct['instance_idx'])-1:
        return dct

    # Change static data accordingly
    pid = 0
    new_instance_idx = [pid]
    for i, (st, en) in enumerate(zip(dct['instance_idx'][:-1], dct['instance_idx'][1:])):
        if i not in ok_ids:
            continue
        pid += en-st
        new_instance_idx.append(pid)

    # old_num_obj_pts = (np.array(dct['instance_idx'])[1:] - np.array(dct['instance_idx'])[:-1]).tolist()
    # new_num_obj_pts = (np.array(new_instance_idx)[1:] - np.array(new_instance_idx)[:-1]).tolist()
    # print("Old num obj pts: {} . New num obj pts {}".format(old_num_obj_pts, new_num_obj_pts))

    imp_keys = ["root_des_radius", "root_num", "clusters", "instance", "material", "obj_points"]
    for key in imp_keys:
        if key in dct:
            dct[key] = [x for i, x in enumerate(dct[key]) if i in ok_ids]

    dct["n_objects"] = len(new_instance_idx)-1
    dct["instance_idx"] = np.array(new_instance_idx)
    dct["ok_ids"] = ok_ids

    # Change dynamic data accordingly
    dct['rot'] = dct['rot'][:, ok_ids]
    dct['trans'] = dct['trans'][:, ok_ids]
    return dct


def subsample_particles_on_large_objects(dct, limit=3000):
    new_instance_idx = [0]
    is_subsample = False
    for oi, (st, en) in enumerate(zip(dct['instance_idx'][:-1], dct['instance_idx'][1:])):
        if en-st > limit and dct["instance"][oi] != b'cloth_square':
            idxs = np.random.choice(en-st, limit, replace=False)
            dct['obj_points'][oi] = dct['obj_points'][oi][idxs]
            dct['clusters'][oi][0][0] = dct['clusters'][oi][0][0][idxs]
            is_subsample = True
        tmp = new_instance_idx[-1] + len(dct['obj_points'][oi])
        new_instance_idx.append(tmp)
    if not is_subsample:
        return dct
    
    # print("subsampled from old instance idx {} to new instance idx {}".format(dct['instance_idx'], new_instance_idx))
    dct["instance_idx"] = new_instance_idx
    return dct


def get_transformation_matrix(rot, trans, scale=None):
    if rot.shape[-1] == 4:
        rotmat = Rotation.from_quat(rot.reshape([-1, 4])).as_matrix().reshape(list(rot.shape[:-1]) + [3,3])
    else:
        rotmat = rot
    if scale is not None:
        rotmat = rotmat * scale[..., None, :]
    ret = np.zeros(list(rotmat.shape[:-2]) + [4,4])
    ret[..., :3, :3] = rotmat
    ret[..., :3, 3] = trans
    ret[..., 3, 3] = 1
    return ret


def transform_points(pts, transform, scale=None, instance_idx=None):
    # pts: O x [V x 3] or OV x 3
    # transform: O x 4 x 4
    # scale: O x 3
    # returns O x [V x 3] or OV x 3

    if transform.ndim == 4:
        ret = [transform_points(pts, tr, scale, instance_idx) for tr in transform]
        if not isinstance(pts, list):
            ret = np.stack(ret)
        return ret

    if instance_idx is not None:
        pts = np.split(pts, instance_idx[1:-1])
    if scale is not None:
        transform = transform.copy()
        transform[:, :3, :3] = transform[:, :3, :3] * scale[:, None, :]
    ret = [ (t[:3,:3] @ p.T).T + t[:3, 3][None] for p, t in zip(pts, transform)]  # O x [V x 3]
    if instance_idx is not None:
        ret = np.concatenate(ret, axis=0)
    return ret  # O x [V x 3] or OV x 3

def reverse_transform_points(pts, transform, scale=None, instance_idx=None):
    if instance_idx is not None:
        pts = np.split(pts, instance_idx[1:-1])
    if instance_idx is not None:
        pts = np.split(pts, instance_idx[1:-1])
    if scale is not None:
        transform = transform.copy()
        transform[:, :3, :3] = transform[:, :3, :3] / scale[:, None, :]
    ret = [ (t[:3,:3].T @ (p - t[:3, 3][None]).T).T for p, t in zip(pts, transform)]
    if instance_idx is not None:
        ret = np.concatenate(ret, axis=0)
    return ret


def get_particle_poses_and_vels_2(dct, noise_std=None):
    """Adds poses and vels of shape (OV x 3) to the dictionary."""
    assert 'cur_transform' in dct and 'prev_transform' in dct
    obj_pts = dct['obj_points']  # O x [V x 3]
    assert np.all(dct['instance_idx'] == np.cumsum([0] + [op.shape[0] for op in obj_pts]))
    assert all([op.shape[0] > 0 for op in obj_pts])

    prev_poses = np.concatenate(transform_points(obj_pts, dct['prev_transform']), 0)  # OV x 3
    cur_poses = np.concatenate(transform_points(obj_pts, dct['cur_transform']), 0)  # OV x 3
    if noise_std is not None:
        noise = np.cumsum(np.random.normal(0, noise_std, [3, cur_poses.shape[0], 3]), 0)
        prev_poses += noise[0]
        cur_poses += noise[1]
    cur_vels = (cur_poses - prev_poses) / dct['dt']  # OV x 3
    dct['cur_poses'], dct['cur_vels'], dct['prev_poses'] = cur_poses, cur_vels, prev_poses

    if 'next_transform' in dct:
        next_poses = np.concatenate(transform_points(obj_pts, dct['next_transform']), 0)  # OV x 3
        if noise_std is not None: next_poses += noise[2]
        next_vels = (next_poses - cur_poses) / dct['dt'] # OV x 3
        dct['next_poses'], dct['next_vels'] = next_poses, next_vels
    
    # transform CoMs
    prev_coms = np.concatenate(transform_points(dct['coms'][:, None, :], dct['prev_transform']), 0) # O x 3
    cur_coms = np.concatenate(transform_points(dct['coms'][:, None, :], dct['cur_transform']), 0) # O x 3
    cur_com_vels = (cur_coms - prev_coms) / dct['dt'] # O x 3
    dct['prev_coms'], dct['cur_coms'], dct['cur_com_vels'] = prev_coms, cur_coms, cur_com_vels
    if 'next_transform' in dct:
        next_coms = np.concatenate(transform_points(dct['coms'][:, None, :], dct['next_transform']), 0) # O x 3
        next_com_vels = (next_coms - cur_coms) / dct['dt'] # O x 3
        dct['next_coms'], dct['next_com_vels'] = next_coms, next_com_vels
    return dct


def find_relations_neighbor(poses, query_idx, anchor_idx, radius, order):
    # poses: OV x 3
    # query_idx: Q
    # anchor_idx: A
    # returns rels: R x 2
    if len(anchor_idx) == 0: return np.zeros((0, 2), dtype=int)
    point_tree = spatial.cKDTree(poses[anchor_idx])
    neighbors = point_tree.query_ball_point(poses[query_idx], radius, p=order)
    relations = []
    for i in range(query_idx.shape[0]):
        if len(neighbors[i]) == 0:
            continue
        receiver = np.ones(len(neighbors[i]), dtype=int) * query_idx[i]
        sender = np.array(anchor_idx[neighbors[i]])
        relations.append(np.stack([receiver, sender], axis=1))
    if not relations:
        return np.zeros((0, 2), dtype=int)
    return np.concatenate(relations, 0)


def find_relations_neighbor_scene(poses, instance_idx, radius, same_obj_rels):
    """Create rels using nearest neighbor search. same_obj_rels: if True, create rels between points in the same object."""
    num_pts = poses.shape[0]
    if same_obj_rels:
        inter_rels = find_relations_neighbor(poses, np.arange(num_pts, dtype=int), np.arange(num_pts, dtype=int), radius, 2)
    else:
        inter_rels = []
        for oi, (st, en) in enumerate(zip(instance_idx[:-1], instance_idx[1:])):
            orels = find_relations_neighbor(poses, np.arange(st, en, dtype=int), 
                np.concatenate([np.arange(0, st, dtype=int), np.arange(en, num_pts, dtype=int)], 0), radius, 2)
            inter_rels.append(orels)
        inter_rels = np.concatenate(inter_rels, 0)
    return inter_rels


def careful_revoxelized(vox, shape):
    """trimesh.voxel.VoxelGrid.revoxelized can miss the boundary. This is hacky modified function to prevent that."""
    shape = tuple(shape)
    bounds = vox.bounds.copy()
    extents = vox.extents
    bounds[0] += extents * 0.01
    bounds[1] -= extents * 0.01
    points = trimesh.util.grid_linspace(
        bounds, shape).reshape(shape + (3,))
    dense = vox.is_filled(points)
    scale = extents / np.asanyarray(shape)
    translate = bounds[0]
    return trimesh.voxel.VoxelGrid(
        dense,
        transform=trimesh.transformations.scale_and_translate(scale, translate))


def get_imp_particles(shape_label, radius, assets, transform, scale, curv_threshold=70):
    # points to canonical meshes
    can_meshes = [assets[sl.item()]['mesh'] for oi, sl in enumerate(shape_label)]
    spacing = 0.1

    mesh_pts = transform_points([ms.vertices.view(np.ndarray) for ms in can_meshes], transform, scale)
    meshes = [trimesh.Trimesh(vertices=pts, faces=can_meshes[oi].faces) for oi, pts in enumerate(mesh_pts)]
    meshes = [ms.copy().subdivide_to_size(spacing*3.5) for ms in meshes]
    num_obj_verts = [ms.vertices.shape[0] for ms in meshes]
    ist_idx = np.cumsum([0] + num_obj_verts)

    curv = [mesh_curvature_trimesh(ms, spacing*2) for ms in meshes]
    is_detail_verts = np.concatenate(curv) > curv_threshold
    
    verts = [ms.vertices.view(np.ndarray) for ms in meshes]
    verts_stack = np.concatenate(verts)
    rels = find_relations_neighbor_scene(verts_stack, ist_idx, radius, same_obj_rels=False).reshape(-1)
    is_prox_verts = np.isin(np.arange(ist_idx[-1], dtype=int), rels)
    is_prox_verts = np.logical_or(is_prox_verts, verts_stack[:, 1] < radius)

    is_imp = np.logical_or(is_prox_verts, is_detail_verts)
    # is_imp = np.random.choice([0, 1], size=is_imp.shape[0], p=[0.95, 0.05]).astype(bool)
    # is_imp = np.random.permutation(is_imp)
    # print('imp ratio: ', is_imp.sum() / is_imp.shape[0])
    is_imp = np.split(is_imp, ist_idx[1:-1])
    imp_verts = [vt[ix] for vt, ix in zip(verts, is_imp)]
    imp_verts = reverse_transform_points(imp_verts, transform)
    return imp_verts


def shape_matching(src, tgt):
    # From 'Meshless Deformations Based on Shape Matching'
    # src, tgt: N x 3
    src_cm = src.mean(0)
    tgt_cm = tgt.mean(0)
    sq = src - src_cm[None]
    tp = tgt - tgt_cm[None]
    diff = np.sqrt(((tp - sq)**2).sum(-1)) / np.sqrt((sq**2).sum(-1)).max()
    if diff.max() < 1.e-2:
        return tgt, (np.array([0., 0., 0., 1.]), tgt_cm - src_cm)
    Apq = tp.T @ sq
    try:
        S = scipy.linalg.sqrtm(Apq.T @ Apq)
    except:
        warnings.warn("Error in Shape Matching. Apq {}, Apq.T @ Apq {}, diff.max() {}".format(Apq, Apq.T @ Apq, diff.max()))
        return tgt, (np.array([0., 0., 0., 1.]), tgt_cm - src_cm)
    try:
        R = Apq @ np.linalg.inv(S)
    except:
        warnings.warn("Error in Shape Matching. Apq {}, Apq.T @ Apq {}, diff.max() {}, S {}".format(Apq, Apq.T @ Apq, diff.max(), S))
        return tgt, (np.array([0., 0., 0., 1.]), tgt_cm - src_cm)
    transform = get_transformation_matrix(R, tgt_cm - R @ src_cm)
    return (R @ sq.T).T + tgt_cm[None], transform


def shape_matching_objects(src, pred, instance_idx):
    assert type(src) == type(pred) == np.ndarray, "src and pred must be numpy arrays"
    assert src.shape == pred.shape, "src and pred must have the same shape"
    assert src.shape[0] == instance_idx[-1], "src and instance_idx should match"
    out = np.zeros_like(pred)
    transforms = []
    for oi, (st, en) in enumerate(zip(instance_idx[:-1], instance_idx[1:])):
        out[st:en], rt = shape_matching(src[st:en], pred[st:en])
        transforms.append(rt)
    return out, np.stack(transforms)


def seq_datapath_to_fish_datapath(filename):
    fish_particle_data_root = '/ccn2/u/chpatel/dpinet_data/data'
    scenario = filename.split('/')[-2]
    seq_name = filename.split('/')[-1].split('.')[0]
    subset = [d for d in ['dynamics_training', 'readout_training', 'model_testing'] if d in filename]
    assert len(subset) == 1, "Unable to determine subset of {}".format(filename)
    subset = subset[0]
    return os.path.join(fish_particle_data_root, subset, scenario, seq_name)


def clip_std(a, n=3):
    m, s = a.mean(), a.std()
    return np.clip(a, m-n*s, m+n*s)

def mesh_curvature(mesh):
    # https://blender.stackexchange.com/questions/146819/is-there-a-way-to-calculate-mean-curvature-of-a-triangular-mesh/147371#147371

    e = mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))                # edges v1,v2
    fa = mesh.face_angles[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))         # face angles at v1,v2

    p1, p2 = mesh.vertices[e[:, 0]], mesh.vertices[e[:, 1]]
    n1, n2 = mesh.vertex_normals[e[:, 0]], mesh.vertex_normals[e[:, 1]]
    pd, nd = (p2-p1), (n2-n1)
    ecurv = (pd*nd).sum(-1) / (pd**2).sum(-1)
    ecurv = np.abs(ecurv)

    curv = np.zeros(mesh.vertices.shape[0])
    denom = np.zeros(mesh.vertices.shape[0])

    for i in [0, 1]:
        np.add.at(curv, e[:, i], ecurv * fa[:, i])
        np.add.at(denom, e[:, i], fa[:, i])

    return curv / denom / 2


def mesh_curvature_trimesh(mesh, radius):
    curv = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius)
    return np.abs(curv)