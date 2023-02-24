# preprocessing
import h5py
import copy
import os
import trimesh
import pickle
import numpy as np
import random
import argparse
import imageio
import scipy
from scipy.spatial.transform import Rotation as R
import copy
import tempfile
import trimesh
import subprocess
from dataset import binvox_rw


dt = 0.01

def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()

def get_bad_meshes_ratio(fpath="./dataset/models_full_check_window.txt"):
    bad_meshes_ratio = dict()
    with open(fpath, "r") as f:
        for line in f:
            obj_name, x_ratio, y_ratio, z_ratio = line.split(",")
            bad_meshes_ratio[obj_name.encode('UTF-8')] = [float(x_ratio), float(y_ratio), float(z_ratio)]
    return bad_meshes_ratio

def correct_roll_name_scale_data(object_names, scales):
    # this is a bug in the tdw_physics data generation for the Roll scenario with ramp
    # the object name and scales are swaped
    if b'ramp_with_platform_30' in object_names:
        if(object_names[2] == b'ramp_with_platform_30'):
            # correct the scales and object_names
            object_name1 = copy.deepcopy(object_names[1])
            object_names[1] = copy.deepcopy(object_names[2])
            object_names[2] = object_name1

            scales1 = copy.deepcopy(scales[1])
            scales[1] = copy.deepcopy(scales[2])
            scales[2] = scales1
        else:
            assert(object_names[1] == b'ramp_with_platform_30')
            assert(np.linalg.norm(scales[1] - np.array([0.2, 1, 0.5])) < 0.000001)
    return object_names, scales

def process_object_mesh(vertices, scale, obj_name, bad_meshes_ratio):

    #checking empty mesh
    if vertices.shape[0] == 0:
        nonexist = 1
        if obj_name in  [b'cloth_square', b'buddah', b'bowl', b'cone', b'cube', b'cylinder', b'dumbbell', b'octahedron', b'pentagon', b'pipe', b'platonic', b'pyramid', b'sphere', b'torus', b'triangular_prism']:
            print("critical object with empty mesh", obj_name)
            print("there should a problem with a data. Please contact the authors to check this")
            import ipdb; ipdb.set_trace()
    else:
        nonexist = 0

        if obj_name in bad_meshes_ratio:
            print("bad mesh ratio", obj_name, scale)
            # cork objects has weird bounding box from tdw like 1.0, 0, 0, so only use the first dimension
            if obj_name in [b"cork_plastic_black", b'tapered_cork_w_hole']:
                scale *= bad_meshes_ratio[obj_name][0]
            else:
                scale *= np.array(bad_meshes_ratio[obj_name])
            print("after", scale)

        if max(scale) > 25:
            # stop if it is not the long bar
            # some heuristic for bad
            if not(abs(scale[0] - 0.05) < 0.00001 and abs(scale[1] - 0.05) < 0.00001 and abs(scale[2] - 100) < 0.00001):
                if not (obj_name == b"889242_mesh" and np.max(abs(scale - 37.092888)) < 0.00001):
                    if obj_name in [b"cork_plastic_black", b'tapered_cork_w_hole']:
                        #pass
                        scale[:] = 25

        vertices[:,0] *= scale[0]
        vertices[:,1] *= scale[1]
        vertices[:,2] *= scale[2]

    return vertices, scale, nonexist


def get_voxelization_params(mesh, spacing):

    edges = mesh.bounding_box.extents
    meshLower = mesh.bounds[0,:]
    meshUpper = mesh.bounds[1,:]

    #  tweak spacing to avoid edge cases for particles laying on the boundary
    # just covers the case where an edge is a whole multiple of the spacing.
    spacingEps = spacing*(1.0 - 1e-4)
    num_vox = np.where(edges < spacing, np.ones(3, dtype=int), (edges/spacingEps).astype(int))
    maxDim = np.max(num_vox)

    #expand border by two voxels to ensure adequate sampling at edges
    # extending by a small offset to avoid point sitting exactly on the boundary
    meshLower_spaced = meshLower - 2.0 * spacing
    meshUpper_spaced = meshUpper +  2.0 * spacing
    maxDim_spaced = maxDim + 4

    # handle big objects
    voxelsize_limit = 512
    if maxDim_spaced > voxelsize_limit:
        for dim in range(3):
            if edges[dim] < (voxelsize_limit - 4) * spacing:
                continue # short edge, no need to chunk
            else:
                amount_to_cut = edges[dim] - (voxelsize_limit - 4) * spacing
                meshLower_spaced[dim] += amount_to_cut * 0.5
                meshUpper_spaced[dim] -= amount_to_cut * 0.5
        maxDim_spaced = voxelsize_limit

    # we shift the voxelization bounds so that the voxel centers
    # lie symmetrically to the center of the object. this reduces the
    # chance of missing features, and also better aligns the particles
    # with the mesh
    # ex. |1|1|1|0.3| --> |0.15|1|1|0.15|
    meshOffset = 0.5 * (spacing - (edges - (num_vox - 1) * spacing))
    meshLower_spaced -= meshOffset
    meshUpper_spaced = meshLower_spaced + maxDim_spaced * spacing

    return maxDim_spaced, meshLower_spaced, meshUpper_spaced


def voxelize(mesh, mesh_path, spacing):
    vox_path = mesh_path.rsplit(".")[0] + ".binvox"
    assert not os.path.exists(vox_path), "File {} already exists".format(vox_path)
    assert not os.path.exists(mesh_path), "File {} already exists".format(mesh_path)
    vox_params = get_voxelization_params(mesh, spacing)
    maxDim_spaced, meshLower_spaced, meshUpper_spaced = vox_params
    trimesh.exchange.export.export_mesh(mesh, mesh_path)
    cmd = f'binvox -aw -dc -d {maxDim_spaced} -bb {meshLower_spaced[0]} {meshLower_spaced[1]} {meshLower_spaced[2]} {meshUpper_spaced[0]} {meshUpper_spaced[1]} {meshUpper_spaced[2]} -t binvox {mesh_path}'
    print(cmd)
    binvox_out = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    if binvox_out.returncode != 0:
        print(binvox_out.stdout)
        print(binvox_out.stderr)
        raise Exception("binvox failed")
    assert os.path.exists(vox_path), "Voxelization failed. File {} does not exist.".format(vox_path)
    return vox_path, vox_params


def voxel_to_particles(voxel_path, spacing, vox_params, bbox=None):
    with open(voxel_path, 'rb') as f:
         m1 = binvox_rw.read_as_3d_array(f)
    maxDim_spaced, meshLower_spaced, meshUpper_spaced = vox_params
    adjusted_spacing = spacing
    x, y, z = np.nonzero(m1.data)
    points = np.expand_dims(meshLower_spaced, 0) + np.stack([(x + 0.5)*adjusted_spacing, (y + 0.5)*adjusted_spacing, (z + 0.5)*adjusted_spacing], axis=1)
    if bbox is not None:
        lower_bound = bbox[0, :]
        upper_bound = bbox[1, :]

        idx = (points[:, 0] - upper_bound[0] <= 0) * (points[:, 0] - lower_bound[0] >= 0)
        idy = (points[:, 1] - upper_bound[1] <= 0) * (points[:, 1] - lower_bound[1] >= 0)
        idz = (points[:, 2] - upper_bound[2] <= 0) * (points[:, 2] - lower_bound[2] >= 0)

        points = points[idx*idy*idz]

    return points


def voxelize_my(mesh, mesh_path, vox_dim):
    vox_path = mesh_path.rsplit(".")[0] + ".binvox"
    assert not os.path.exists(vox_path), "File {} already exists".format(vox_path)
    assert not os.path.exists(mesh_path), "File {} already exists".format(mesh_path)

    trimesh.exchange.export.export_mesh(mesh, mesh_path)  # for viz help
    kwargs = {'use_offscreen_pbuffer': False, 'dilated_carving': True, 'wireframe':True, 'dimension': vox_dim}
    vox = mesh.voxelized(pitch=None, method='binvox', **kwargs)
    return vox

def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))  #edit
    lens = [len(a) for a in arrs]
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz*=s
    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)
    return tuple(ans)

def downsample_vox(vox, factor):
    arrs = [np.round(np.arange(0, s)/f).astype(int) for s, f in zip(vox.shape, factor)]
    coords = meshgrid2(*arrs)
    coords = tuple(c.flatten() for c in coords)
    vals = vox.matrix.flatten()

    tgt_dim = max(c.max() for c in coords) + 1
    new_occ = np.zeros([tgt_dim]*3, dtype=int)
    np.add.at(new_occ, coords, vals)
    new_occ = new_occ > 0

    ntr = vox.transform.copy()
    ntr[:3, :3] *= factor[:, None]
    return trimesh.voxel.VoxelGrid(new_occ, transform=ntr)


def voxel_to_particles_my(vox, spacing):
    factor = spacing/vox.pitch
    dvox = downsample_vox(vox, factor)
    return dvox.points


def process_redyellow_id(object_ids, yellow_id, red_id, nonexists):

    yellow_id_order = [order_id for order_id, id_ in enumerate(object_ids) if id_ == yellow_id]
    assert(len(yellow_id_order) == 1)
    yellow_id_order = yellow_id_order[0]

    red_id_order = [order_id for order_id, id_ in enumerate(object_ids)  if id_ == red_id]
    assert(len(red_id_order) == 1)
    red_id_order = red_id_order[0]

    assert(yellow_id_order==0)

    safe_up_to_idx = max(yellow_id_order, red_id_order)
    for id_ in range(safe_up_to_idx):
        assert(not nonexists[id_])
    return yellow_id_order, red_id_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--method', type=str, default='fish')
    parser.add_argument('--vox_dim', type=int, default=None)
    parser.add_argument('--spacing', type=float, default=0.05)
    args = parser.parse_args()
    assert args.method in ['fish', 'my']
    if args.method == 'my': assert args.vox_dim is not None

    bad_meshes_ratio = get_bad_meshes_ratio()
    # tmp_path = tempfile.mkdtemp()

    scenario = args.scenario
    flex_engine = ("Drape" in scenario)

    seq_names = [file.replace('.hdf5', '') for file in os.listdir(os.path.join(args.data_dir, scenario)) if file.endswith("hdf5")]
    seq_names.sort()

    for sid, seq_name in enumerate(seq_names):
        seq_out_dir = os.path.join(args.out_dir, scenario, seq_name)
        os.makedirs(seq_out_dir, exist_ok=False)
        filename = os.path.join(args.data_dir, scenario, seq_name + '.hdf5')
        print("Processing ({}/{}) {}".format(sid, len(seq_names), filename))
        f = h5py.File(filename, "r")

        object_ids = f["static"]["object_ids"][:].tolist()
        scales = f["static"]["scale"][:]
        object_names = f["static"]["model_names"][:]
        print("model_names", f["static"]["model_names"][:])

        object_names, scales = correct_roll_name_scale_data(object_names, scales)

        nonexists = np.zeros((len(object_ids)), dtype=bool)
        clusters = []
        root_num = []
        root_des_radius = []
        instance_idx = [0]
        instance = []
        material = []
        obj_points = []

        for idx, object_id in enumerate(object_ids):
            obj_name = f["static"]["model_names"][:][idx]
            obj_mat = 'cloth' if obj_name == b'cloth_square' else 'rigid'

            if flex_engine:
                points = f["frames"]["0000"]["particles"][str(object_id)][:]
                obj_points.append(points)

            else:
                #vertices, faces = self.object_meshes[object_id]
                vertices = f["static"]["mesh"][f"vertices_{idx}"][:]
                faces = f["static"]["mesh"][f"faces_{idx}"][:]

                vertices, scales[idx], nonexists[idx] = process_object_mesh(vertices, scales[idx].copy(), obj_name, bad_meshes_ratio)

                if nonexists[idx] == 1:
                    continue

                obj_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
                mesh_path = os.path.join(seq_out_dir, f"{idx}.obj")
                if args.method == 'fish':
                    vox_path, vox_params = voxelize(obj_mesh, mesh_path=mesh_path, spacing=args.spacing)
                    points = voxel_to_particles(vox_path, spacing=args.spacing, vox_params=vox_params, bbox=obj_mesh.bounds)
                elif args.method == 'my':
                    vox = voxelize_my(obj_mesh, mesh_path=mesh_path, vox_dim=args.vox_dim)
                    points = voxel_to_particles_my(vox, spacing=args.spacing)

                if points.shape[0] == 0:
                    nonexists[idx] = 1
                    continue
                obj_points.append(points)

            npts = points.shape[0]
            instance_idx.append(instance_idx[-1] + npts)
            clusters.append([[np.array([0]* npts, dtype=np.int32)]])
            root_num.append([1])
            root_des_radius.append([args.spacing])
            instance.append(obj_name)
            material.append(obj_mat)

        nsteps = len(f["frames"])
        n_objects = len(obj_points)
        n_particles = np.sum([obj_pts.shape[0] for obj_pts in obj_points])

        yellow_id = f["static"]["zone_id"][()]
        red_id = f["static"]["target_id"][()]
        yellow_id_order, red_id_order = process_redyellow_id(object_ids, yellow_id, red_id, nonexists)


        phases_dict = dict()
        phases_dict["instance_idx"] = instance_idx
        phases_dict["root_des_radius"] = root_des_radius
        phases_dict["root_num"] = root_num
        phases_dict["clusters"] = clusters
        phases_dict["instance"] = instance
        phases_dict["material"] = material
        phases_dict["time_step"] = nsteps
        phases_dict["n_objects"] = n_objects
        phases_dict["n_particles"] = n_particles
        phases_dict["obj_points"] = obj_points
        phases_dict["dt"] = dt
        phases_dict["yellow_id"] = yellow_id_order
        phases_dict["red_id"] = red_id_order

        assert(phases_dict["n_objects"] == len(phases_dict["instance_idx"]) - 1)
        assert(phases_dict["n_objects"] == len(phases_dict["root_des_radius"]))
        assert(phases_dict["n_objects"] == len(phases_dict["root_num"]))
        assert(phases_dict["n_objects"] == len(phases_dict["clusters"]))
        assert(phases_dict["n_objects"] == len(phases_dict["instance"]))
        assert(phases_dict["n_objects"] == len(phases_dict["material"]))
        assert(phases_dict["n_objects"] == len(phases_dict["obj_points"]))
        for obj_pts in obj_points:# check not empty mesh
            assert(obj_pts.shape[0] > 0)

        with open(os.path.join(seq_out_dir, 'phases_dict.pkl'), "wb") as pf:
            pickle.dump(phases_dict, pf)



        if flex_engine:
            seq_poses = []
            seq_vels = []
            for step in range(nsteps):
                positions = []
                velocities = []
                for idx, object_id in enumerate(object_ids):
                    pos = f["frames"][f"{step:04}"]["particles"][str(object_id)][:][:,:3]
                    vel = f["frames"][f"{step:04}"]["velocities"][str(object_id)][:][:,:3]
                    positions.append(pos)
                    velocities.append(vel)
                seq_poses.append(np.concatenate(positions, axis=0))
                seq_vels.append(np.concatenate(velocities, axis=0))
            seq_poses = np.stack(seq_poses)
            seq_vels = np.stack(seq_vels)
            assert(n_objects == seq_poses.shape[1]), f"nobjects does not match number of positions: {n_objects} vs {positions.shape[0]}"
            store_data(['particle_positions', "particle_velocities"], [seq_poses, seq_vels], os.path.join(seq_out_dir, 'dynamic_data.h5'))

        else:
            seq_poses = []
            seq_rots = []
            for step in range(nsteps):
                positions = f["frames"][f"{step:04}"]["objects"]["positions"][:]
                rotations = f["frames"][f"{step:04}"]["objects"]["rotations"][:] #x,y,z,w

                if np.sum(nonexists) > 0:
                    #remove the missing guy
                    pos_ = []
                    ros_ = []
                    for idx, object_id in enumerate(object_ids):
                        if not nonexists[idx]:
                            pos_.append(positions[idx])
                            ros_.append(rotations[idx])
                    positions = np.stack(pos_, axis=0)
                    rotations = np.stack(ros_, axis=0)
                seq_poses.append(positions)
                seq_rots.append(rotations)

            seq_poses = np.stack(seq_poses)
            seq_rots = np.stack(seq_rots)

            assert(seq_poses.shape[1] == phases_dict["n_objects"]), f"nobjects does not match number of positions: {n_objects} vs {positions.shape[0]}"
            store_data(['obj_positions', 'obj_rotations'], [seq_poses, seq_rots], os.path.join(seq_out_dir, 'dynamic_data.h5'))


if __name__ == '__main__':
    main()
