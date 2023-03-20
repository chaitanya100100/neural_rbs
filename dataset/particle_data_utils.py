import numpy as np
from scipy.spatial.transform.rotation import Rotation
import scipy.spatial as spatial
import trimesh
import warnings
import scipy


def get_transformation_matrix(rot, trans, scale=None):
    """Get 4x4 transformation matrix from rotation, translation and optionally scale. Works with batched input.
    Args:
        rot: N x 4 quaternion or N x 3 x 3 rotation matrix
        trans: N x 3 translation
        scale: N x 3 scale
    Returns:
        N x 4 x 4 transformation matrix
    """
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
    """Transform scene points by a transformation matrix.
    O is object dimension. V is point dimension. Also works with batched input.
    Args:
        pts: O x [V x 3] or OV x 3
        transform: O x 4 x 4
        scale: O x 3
        instance_idx: O+1 lengthed array denoting start and end indices of each object.
            Required if pts is OV x 3
    Returns:
        O x [V x 3] or OV x 3
    """
    # Batched input
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


def get_particle_poses_and_vels_2(dct, noise_std=None):
    """Adds poses and vels of shape (OV x 3) to the data dictionary.
    
    It will calculate cur_poses, cur_vels, prev_poses, cur_coms, cur_com_vels, prev_coms.
    If next_transform is in the dictionary, it will also calculate next_poses, next_vels, next_coms, next_com_vels.
    """
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
    """Find nearest neighbors within a radius.
    Args:
        poses: OV x 3 array of points
        query_idx: Q array of indices to query
        anchor_idx: A array of indices to consider for neighbors
        order: p-norm to use for distance
    Returns:
        rels: R x 2 array of relations (receiver, sender)
    """
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
    """A useful wrapper around `find_relations_neighbor`.
    Args:
        poses: OV x 3 array of points
        instance_idx: O+1 array of indices to separate objects.
        radius: radius to use for neighbor search
        same_obj_rels: whether to include relations between points in the same object
    Returns:
        inter_rels: R x 2 array of relations (receiver, sender) between objects.
    """
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


def shape_matching(src, tgt):
    """Shape matching between two point clouds.
    From the paper 'Meshless Deformations Based on Shape Matching'.
    src, tgt: N x 3
    """
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
    """Shape matching between two frames. A wrapper around `shape_matching` for every object.
    Args:
        src: OV x 3 array of points. source rigid object.
        pred: OV x 3 array of points. predicted points with possible non-rigidity.
        instance_idx: O+1 array of indices to separate objects.
    Returns:
        out: OV x 3 array of points. predicted points with rigidity.
        transforms: O x 4 x 4 array of transformations for each object from src to pred.
    """
    assert type(src) == type(pred) == np.ndarray, "src and pred must be numpy arrays"
    assert src.shape == pred.shape, "src and pred must have the same shape"
    assert src.shape[0] == instance_idx[-1], "src and instance_idx should match"
    out = np.zeros_like(pred)
    transforms = []
    for oi, (st, en) in enumerate(zip(instance_idx[:-1], instance_idx[1:])):
        out[st:en], rt = shape_matching(src[st:en], pred[st:en])
        transforms.append(rt)
    return out, np.stack(transforms)
