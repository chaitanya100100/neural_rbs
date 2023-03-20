import numpy as np
import torch
import torch.nn
import h5py


def recursive_apply(x, func):
    """Apply func to all elements of x. Works recursively on nested lists and dicts."""
    if isinstance(x, dict):
        return {k: recursive_apply(v, func) for k, v in x.items()}
    elif isinstance(x, list):
        return [recursive_apply(i, func) for i in x]
    else:
        return func(x)

def recursive_to_tensor(x):
    """Convert all elements of x to torch tensors. Works recursively on nested lists and dicts."""
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
    """Convert all elements from pytorch tensors to numpy arrays. Works recursively on nested lists and dicts."""
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

