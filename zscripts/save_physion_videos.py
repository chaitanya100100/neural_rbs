import numpy as np
import h5py
import os
from os.path import join
import glob
from tqdm.notebook import tqdm
from multiprocessing import Pool, cpu_count

from dataset import data_utils
from utils.viz_utils import save_video
from dataset.data_utils import SCENARIOS

def save_physion_video(inp_path, out_path):

    with h5py.File(inp_path, 'r') as h5:
        num_frames = data_utils.get_num_frames(h5)
        imgs, segs = data_utils.index_imgs(h5, np.arange(num_frames))
    save_video(imgs, out_path, 30)


def main():

    data_dir = "/ccn2/u/rmvenkat/chpatel/physion_dataset/model_testing/"
    out_dir = "/ccn2/u/rmvenkat/chpatel/data_viz/"
    args = []
    for scen in SCENARIOS:
        print(scen)
        for fpath in glob.glob(join(data_dir, scen, '*.hdf5')):
            args.append([fpath, join(out_dir, scen, os.path.basename(fpath).split('.')[0]+".webm" )])

    # args = args[:100]
    print(len(args))

    pool = Pool(processes=int(cpu_count()*0.6))
    for a in args:
        pool.apply_async(save_physion_video, args=a)
    pool.close()
    pool.join()
    print("Done")

if __name__ == '__main__':
    main()