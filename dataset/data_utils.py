import io
import numpy as np
import torch
from PIL import Image
import torch.nn
import cv2
import glob
import os
import json
import pickle as pkl
import h5py


SCENARIOS = [
    'Collide',
    'Dominoes',
    'Drop',
    'Roll',
    'Contain',
    # 'Drape',
    'Link',
    'Support',
]

READOUT_MAXOBJ = {
    'Collide': 7,
    'Dominoes': 11,
    'Drop': 7,
    'Roll': 7,
    'Contain': 10,
    'Drape': 6,
    'Link': 15,
    'Support': 9,
}

READOUT_THRESHOLD = {
    'Collide': 1.5,
    'Dominoes': 1.5,
    'Drop': 1,
    'Roll': 1.5,
    'Contain': 1,
    'Link': 1.5,
    'Support': 1.5,
}

MAX_ROLLOUT_LENGTH = {
    'Collide': 165,
    'Dominoes': 315,
    'Drop': 315,
    'Roll': 315,
    'Contain': 375,
    'Drape': 165,
    'Link': 420,
    'Support': 615,
}


def get_num_frames(h5_file):
    return len(h5_file['frames'].keys())

def get_image(raw_img):
    img = Image.open(io.BytesIO(raw_img))
    return np.array(img)

def get_depth_values(image: np.array, depth_pass: str = "_depth", width: int = 256, height: int = 256, near_plane: float = 0.1, far_plane: float = 100) -> np.array:
    """
    Get the depth values of each pixel in a _depth image pass.
    The far plane is hardcoded as 100. The near plane is hardcoded as 0.1.
    (This is due to how the depth shader is implemented.)
    :param image: The image pass as a numpy array.
    :param depth_pass: The type of depth pass. This determines how the values are decoded. Options: `"_depth"`, `"_depth_simple"`.
    :param width: The width of the screen in pixels. See output data `Images.get_width()`.
    :param height: The height of the screen in pixels. See output data `Images.get_height()`.
    :param near_plane: The near clipping plane. See command `set_camera_clipping_planes`. The default value in this function is the default value of the near clipping plane.
    :param far_plane: The far clipping plane. See command `set_camera_clipping_planes`. The default value in this function is the default value of the far clipping plane.
    :return An array of depth values.
    """

    # Convert the image to a 2D image array.
    image = np.flip(np.reshape(image, (height, width, 3)), 0)
    if depth_pass == "_depth":
        depth_values = np.array((image[:, :, 0] + image[:, :, 1] / 256.0 + image[:, :, 2] / (256.0 ** 2)))
    elif depth_pass == "_depth_simple":
        depth_values = image[:, :, 0] / 256.0
    else:
        raise Exception(f"Invalid depth pass: {depth_pass}")
    # Un-normalize the depth values.
    return (depth_values * ((far_plane - near_plane) / 256.0)).astype(np.float32)


def index_img(h5_file, index, get_depth=False):
    img0 = h5_file['frames'][str(index).zfill(4)]['images']
    suffix = '' if '_img' in img0.keys() else '_cam0'
    rgb_img = get_image(img0['_img' + suffix][:])
    segments = get_image(img0['_id' + suffix][:])
    if get_depth:
        wd = rgb_img.shape[-2]
        depth = get_depth_values(img0['_depth' + suffix][:], width=wd, height=wd)
    else:
        depth = None
    return rgb_img, segments, depth

def index_imgs(h5_file, indices, scale_down_test_img=True, get_depth=False):
    """Get image and segmentation of specified indices."""
    all_imgs = []
    all_segs = []
    all_depths = []
    for ct, index in enumerate(indices):
        rgb_img, segments, depth = index_img(h5_file, index, get_depth)
        if scale_down_test_img:
            if rgb_img.shape[0] == 512:  # identify test image
                rgb_img = rgb_img[::2, ::2] 
                segments = segments[::2, ::2]
                if get_depth: depth = depth[::2, ::2]
            assert rgb_img.shape[0] == 256 and rgb_img.shape[1] == 256 
            assert segments.shape[0] == 256 and segments.shape[1] == 256 
            if get_depth: assert depth.shape[0] == 256 and depth.shape[1] == 256 
        all_imgs.append(rgb_img)
        all_segs.append(segments)
        if get_depth: all_depths.append(depth)
    
    all_imgs = np.stack(all_imgs, 0)
    all_segs = np.stack(all_segs, 0)
    
    if get_depth:
        all_depths = np.stack(all_depths, 0)
        return all_imgs, all_segs, all_depths
    return all_imgs, all_segs


def get_object_masks(seg_imgs, seg_colors, background=True):
    # if len(seg_imgs.shape) == 3:
    #     seg_imgs = np.expand_dims(seg_imgs, 0)
    #     is_batch = False
    # else:
    #     is_batch = True

    obj_masks = []
    for scol in seg_colors:
        mask = (seg_imgs == scol)#.astype(np.float)

        # If object is not visible in the frame
        # if mask.sum() == 0:
        #     mask[:] = 1
        obj_masks.append(mask)

    obj_masks = np.stack(obj_masks, 0)    
    obj_masks = obj_masks.min(axis=-1)  # collapse last channel dim

    if background:
        bg_mask = ~ obj_masks.max(0, keepdims=True)
        obj_masks = np.concatenate([obj_masks, bg_mask], 0)
    obj_masks = np.expand_dims(obj_masks, -3)  # make N x 1 x H x W

    # if not is_batch:
    #     obj_masks = obj_masks[0]
    return obj_masks


def get_obj_feat_mean_std(obj_features):
    # obj_features is npz object {<seq_filename>: np array of size N x O x d }
    stacked_feats = []
    for k in obj_features.keys():
        feat = obj_features[k]
        feat = feat.reshape([-1, feat.shape[-1]])
        stacked_feats.append(feat)

    stacked_feats = np.concatenate(stacked_feats)
    mean = stacked_feats.mean(0)
    std = stacked_feats.std(0)
    return mean, std


def _get_split_paths(paths, split, split_percentage):
    if split_percentage is None: return paths
    rng = np.random.RandomState(652)
    rng.shuffle(paths)
    num_files = len(paths)
    at = int(split_percentage * num_files)
    train_paths = paths[:at]
    val_paths = paths[at:]    
    if split == 'train': return train_paths
    elif split == 'val': return val_paths
    else: raise AttributeError("Wrong split")


def get_readout_seq_paths(physion_path, scen, split=None, split_percentage=None):
    if split == 'test':
        assert split_percentage is None
        all_hdf5 = glob.glob(os.path.join(physion_path, 'model_testing', scen, '*.hdf5'))
    elif split in ['train', 'val']:
        assert split_percentage is not None
        all_hdf5 = glob.glob(os.path.join(physion_path, 'readout_training', scen, '*.hdf5'))
        all_hdf5 = _get_split_paths(all_hdf5, split, split_percentage)
    elif split is None:
        assert split_percentage is None
        all_hdf5 = glob.glob(os.path.join(physion_path, 'readout_training', scen, '*.hdf5'))
        all_hdf5 += glob.glob(os.path.join(physion_path, 'model_testing', scen, '*.hdf5'))
    return all_hdf5


def get_dynamics_seq_paths(physion_path, protocol, split=None, split_percentage=None, depth=False):
    """Returns dynamics sequence paths depending on protocol (all, allbut_Roll, only_Roll) and split."""
    is_new_data = physion_path.startswith('/ccn2/u/rmvenkat/data/testing_physion/all_sce_all_fixed/')
    if is_new_data:
        assert depth is False
    else:
        physion_path = os.path.join(physion_path, 'dynamics_training')
        if depth: physion_path += '_depth'

    if protocol.startswith('only_'):
        scen = protocol.split('_')[1]
        assert scen in SCENARIOS
        scenarios = [scen]
    elif protocol.startswith('allbut_'):
        scen = protocol.split('_')[1]
        assert scen in SCENARIOS
        split_percentage = None
        if split == 'train': scenarios = [s for s in SCENARIOS if s != scen]
        elif split == 'val': scenarios = [scen]
        elif split is None: scenarios = SCENARIOS
        else: raise AttributeError("Wrong split")
    elif protocol == 'all':
        scenarios = SCENARIOS
    else:
        raise AttributeError("Wrong protocol")
    
    all_hdf5 = []
    for scen in scenarios:
        if is_new_data:
            res = glob.glob(os.path.join(physion_path, "{}_all_movies".format(scen.lower()), 'train', '*/*.hdf5'))
        else:
            res = glob.glob(os.path.join(physion_path, scen, '*.hdf5'))
            if depth: res = glob.glob(os.path.join(physion_path, scen, '*/*.hdf5'))
        res = _get_split_paths(res, split, split_percentage)
        all_hdf5.extend(res)

    return all_hdf5


def recursive_apply(x, func):
    if isinstance(x, dict):
        return {k: recursive_apply(v, func) for k, v in x.items()}
    elif isinstance(x, list):
        return [recursive_apply(i, func) for i in x]
    else:
        return func(x)

def recursive_to_tensor(x):
    def func(x):
        if isinstance(x, str):
            return x
        if callable(x):
            return x
        if hasattr(x, 'prep_graph') and callable(x.prep_graph):
            return x
        if isinstance(x, np.ndarray):
            if x.dtype.type != np.str_:
                x = torch.from_numpy(x)
        else:
            x = torch.tensor(x)
        if x.dtype == torch.double:
            x = x.float()
        return x
    return recursive_apply(x, func)

def recursive_to_numpy(x):
    def func(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x
    return recursive_apply(x, func)


def collate_fn(data):
    # if particle dataset, return first and only item as batch
    assert len(data) == 1
    return data[0]


def store_hdf5_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for dn, d in zip(data_names, data):
        hf.create_dataset(dn, data=d)
    hf.close()


def nested_shallow_dict(dct, delim='/'):
    ret = {}
    for k, v in dct.items():
        assert delim not in k, "key {} contains delimeter {}".format(k, delim)
        if not isinstance(v, dict):
            ret[k] = v
            continue
        vdct = nested_shallow_dict(v, delim)
        for xk, xv in vdct.items():
            ret[k+delim+xk] = xv
    return ret


def process_movi_tfds_example(example):
    new_example = {}
    for k, v in example.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        if k in ['instances/bboxes', 'instances/bbox_frames']:
            v = v.to_tensor().numpy()
        new_example[k] = v
    return new_example


def load_entire_hdf5(dct):
    if isinstance(dct, h5py.Dataset):
        return dct[()]
    ret = {}
    for k, v in dct.items():
        ret[k] = load_entire_hdf5(v)
    return ret


class ReadoutLabels():
    """An easy way to get OCP labels of physion sequences."""
    def __init__(self, saved_data_path):
        labels = {}
        for scen in SCENARIOS:
            fpath = '{}/ocp_labels_{}.json'.format(saved_data_path, scen)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    labels.update(json.load(f))
        self.labels = labels
        self.subsets = [
            'readout_training',
            'model_testing',
            'dynamics_training',
        ]
    def get(self, key, subset=None):
        if '/' in key:
            seq_name = os.path.basename(key)
            for ss in self.subsets:
                if ss in key:
                    subset = ss
                    break
        else:
            seq_name = key

        if not seq_name.endswith('.hdf5'):
            seq_name += '.hdf5'

        if subset is not None:
            return self.labels[subset + '/' + seq_name]

        for ss in self.subsets:
            if ss + '/' + seq_name in self.labels:
                assert all(nss + '/' + seq_name not in self.labels for nss in self.subsets if ss != nss), "Provided key is ambiguous"
                return self.labels[ss + '/' + seq_name]
        raise KeyError("Key {} not found in labels".format(key))


def merge_particle_dynamics_data():
    # Data of each frame is in a separate file. Merge it into a single file. Need to run it only once.
    particle_data_path = '/ccn2/u/chpatel/dpinet_data/data/dynamics_training/Dominoes/'
    for i, seq_dir in enumerate(glob.glob(particle_data_path + '/*')):
        print(i, seq_dir)
        with open(seq_dir + '/phases_dict.pkl', "rb") as fp:
            dct = pkl.load(fp)
        num_frames = dct['time_step']
        
        trans = []
        rot = []
        for fi in range(num_frames):
            with h5py.File(seq_dir + f'/{fi}.h5', 'r') as h5:
                trans.append(h5['obj_positions'][:])
                rot.append(h5['obj_rotations'][:])
        trans = np.stack(trans)
        rot = np.stack(rot)
        with h5py.File(seq_dir + '/dynamic_data.h5', 'w') as h5:
            h5.create_dataset('obj_positions', data=trans)
            h5.create_dataset('obj_rotations', data=rot)
