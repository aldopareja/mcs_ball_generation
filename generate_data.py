import machine_common_sense as mcs
import numpy as onp
from jax import numpy as jnp
from jax.random import split, PRNGKey
import random
import jax
import h5py
from pathlib import Path
from itertools import starmap
import sys
from pathlib import Path


def get_camera_intrinsics(aspect_ratio, field_of_view):
    width, height = aspect_ratio
    aspect_ratio = width / height

    cx, cy = width / 2.0, height / 2.0

    fov_y = jnp.deg2rad(field_of_view)

    fov_x = 2 * jnp.arctan(aspect_ratio * jnp.tan(fov_y / 2.0))

    fx = cx / jnp.tan(fov_x / 2.0)
    fy = cy / jnp.tan(fov_y / 2.0)

    return cx, cy, fx, fy


def depth_image_to_point_cloud(depth, cx, cy, fx, fy):
    height, width = depth.shape
    xs = (depth * (jnp.arange(width) - cx - 0.5).reshape(1, -1) / fx).reshape(-1)
    ys = (depth * (jnp.arange(height) - cy - 0.5).reshape(-1, 1) / fy).reshape(-1)
    zs = depth.reshape(-1)

    point_cloud = jnp.stack([xs, ys, zs], axis=0)

    return point_cloud


def mask_rgb_to_id(mask_rgb):
    r, g, b = mask_rgb
    return jnp.uint32(r + 1) + jnp.uint32(g + 1) * 256 + jnp.uint32(b + 1) * 256**2


def get_id_from_masks(ball_color, mask_rgb):
    return jax.lax.cond(
        jnp.alltrue(ball_color == mask_rgb),
        lambda: jnp.uint32(0),
        lambda: mask_rgb_to_id(mask_rgb),
    )


@jax.jit
def make_rgb_point_cloud_ball(
    ball_mask_rgb: jnp.ndarray,
    mask: jnp.ndarray,
    rgb: jnp.ndarray,
    depth: jnp.ndarray,
    aspect_ratio,
    field_of_view,
    voxel_res=None,
):
    """computes a flat point cloud (3xN where N is the number of pixels) given a depth map. Also returns the ids, where the ball is always 0 and all others are
    a random integer. It also returns a corresponding RGB cloud.
    """
    cx, cy, fx, fy = get_camera_intrinsics(aspect_ratio, field_of_view)
    depth_cloud = depth_image_to_point_cloud(depth, cx, cy, fx, fy)
    rgb_cloud = jnp.stack([rgb[:, :, i].reshape(-1) for i in range(3)], axis=0)

    ids = jax.vmap(jax.vmap(get_id_from_masks, in_axes=(None, 0)), in_axes=(None, 0))(
        ball_mask_rgb, mask
    ).reshape(-1)

    if voxel_res is not None:
        depth_cloud = jnp.round(depth_cloud / voxel_res) * voxel_res

    return depth_cloud, rgb_cloud, ids


def filter_unique_depths(depth_cloud, rgb_cloud, all_masks):
    _, idxs = jnp.unique(depth_cloud, axis=1, return_index=True)
    return depth_cloud[:, idxs], rgb_cloud[:, idxs], all_masks[idxs]


def load_scene(pr,i, scenes_folder):
    p = scenes_folder / f"pr_{pr}_{i+1:06}.json"
    if not p.exists():
        return None, None, False, p
    scene_data = mcs.load_scene_json_file(str(p))
    ball_id = scene_data["goal"]["metadata"]["target"]["id"]
    s = controller.start_scene(scene_data)
    b = [o for o in s.segmentation_colors if o["objectId"] == ball_id][0]
    ball_color = jnp.array((b["r"], b["g"], b["b"]), dtype=jnp.uint8)
    return ball_color, s, True, p


@jax.jit
def soccer_ball_in_ids(ids):
    return jax.lax.cond(jnp.any(ids == 0), lambda: True, lambda: False)


@jax.jit
def get_ball_centroid(depth, ids):
    pc_sum = jnp.where(ids == 0, depth[3], 0).sum()
    mask_size = (ids == 0).sum()
    return pc_sum / mask_size


def add_random_camera_jitter(key, d, r, ids):
    """adds random rotations to the camera"""
    INV_MOVES = {
        "LookUp": "LookDown",
        "LookDown": "LookUp",
        "RotateLeft": "RotateRight",
        "RotateRight": "RotateLeft",
    }
    vertical_move = random.choice(["LookUp", "LookDown"])
    horizontal_move = random.choice(["RotateLeft", "RotateRight"])
    num_steps = jax.random.randint(key, (1,), minval=0, maxval=4).item()
    for _ in range(num_steps):
        move = random.choice([vertical_move, horizontal_move])
        s = controller.step(move)
        d, r, ids = make_rgb_point_cloud_ball(
            ball_color,
            jnp.asarray(s.object_mask_list[-1]),
            jnp.asarray(s.image_list[-1]),
            jnp.asarray(s.depth_map_list[-1]),
            jnp.array(s.camera_aspect_ratio),
            s.camera_field_of_view,
            voxel_res=0.04,
        )
        if not soccer_ball_in_ids(ids):
            move = INV_MOVES[move]
            s = controller.step(move)
            d, r, ids = make_rgb_point_cloud_ball(
                ball_color,
                jnp.asarray(s.object_mask_list[-1]),
                jnp.asarray(s.image_list[-1]),
                jnp.asarray(s.depth_map_list[-1]),
                jnp.array(s.camera_aspect_ratio),
                s.camera_field_of_view,
                voxel_res=0.04,
            )
            assert soccer_ball_in_ids(ids)
            print("hit view border", num_steps)
            break
    print("num_steps", num_steps)
    return d, r, ids


@jax.jit
def make_centered_padded_data(d_masked, r_masked):
    size = d_masked.shape[1]
    d_centered = d_masked - d_masked.mean(axis=1)[:, None]

    data_mask = jnp.zeros((MAX_SIZE,), dtype=jnp.uint8)
    data_mask = data_mask.at[0:size].set(True)

    data_depth = jnp.zeros((3,MAX_SIZE,))
    data_depth = data_depth.at[:, 0:size].set(d_centered)

    data_rgb = jnp.zeros((3,MAX_SIZE,),dtype=jnp.uint8,)
    data_rgb = data_depth.at[:, 0:size].set(r_masked)
    return data_depth, data_rgb, data_mask


def create_datasets(h5py_file):
    f = h5py_file
    if 'masks' in f:
        masks, rgb_clouds, depth_clouds, labels = map(lambda x: f[x], ['masks', 'rgb_clouds', 'depth_clouds', 'labels'])
        offset = masks.shape[0]
        assert len(onp.unique(onp.array([ x.shape[0] for x in [masks, rgb_clouds, depth_clouds, labels]]))) == 1
    else:
        masks = f.create_dataset(
            "masks",
            (10**1, MAX_SIZE),
            maxshape=(None, MAX_SIZE),
            dtype="i1",
            chunks=(10**2, MAX_SIZE),
        )
        rgb_clouds = f.create_dataset(
            "rgb_clouds",
            (10**1, 3, MAX_SIZE),
            maxshape=(None, 3, MAX_SIZE),
            dtype="i1",
            chunks=(10**2, 3, MAX_SIZE),
        )
        depth_clouds = f.create_dataset(
            "depth_clouds",
            (10**1, 3, MAX_SIZE),
            maxshape=(None, 3, MAX_SIZE),
            dtype="f4",
            chunks=(10**2, 3, MAX_SIZE),
        )
        labels = f.create_dataset(
            "labels", (10**1,), maxshape=(None,), dtype="i1", chunks=(10**2,)
        )
        offset = 0
    return masks, rgb_clouds, depth_clouds, labels, offset


def resize_databases(num_extra):
    list(map(
        lambda x: x.resize(x.shape[0] + num_extra, axis=0),
        [depth_db, rgb_db, m_db, labels_db],
    ))


def save_data(depth, rgb, mask, label, save_i):
    if save_i >= depth_db.shape[0]:
        resize_databases(num_extra=20)

    list(starmap(
        lambda db, val: db.__setitem__(save_i, val),
        zip([depth_db, rgb_db, m_db, labels_db], [depth, rgb, mask, label]),
    ))

    save_i += 1
    return save_i


if __name__ == "__main__":
    MAX_SIZE = 2000
    rank = int(sys.argv[1])
    data_save_folder = Path(sys.argv[2])
    scenes_jsons_folder = Path(sys.argv[3])
    print("rank", rank)
    
    data_path = data_save_folder/ f'soccer_balls_data_{rank}.h5'
    print("data_path", data_path)

    k = PRNGKey(156 + rank)
    controller = mcs.create_controller("sample_config.ini")
    with h5py.File(data_path, "a", libver='latest') as f:
        m_db, rgb_db, depth_db, labels_db, offset = create_datasets(f)
        f.swmr_mode = True
        
        save_i = 0 + offset
        
        for i in range(9999999):
            for pr in [rank,]:
                ball_color, s, file_exists, file_path = load_scene(pr,i, scenes_jsons_folder)
                
                if not file_exists:
                    # print("file not found", file_path)
                    continue
                print(file_path)
                d, r, ids = make_rgb_point_cloud_ball(
                    ball_color,
                    jnp.asarray(s.object_mask_list[-1]),
                    jnp.asarray(s.image_list[-1]),
                    jnp.asarray(s.depth_map_list[-1]),
                    jnp.array(s.camera_aspect_ratio),
                    s.camera_field_of_view,
                    voxel_res=0.04,
                )
                if not soccer_ball_in_ids(ids):
                    print("scene_no_ball", i)
                    file_path.unlink()
                    continue

                k, sk = split(k)
                d, r, ids = add_random_camera_jitter(sk, d, r, ids)

                v_d, v_r, v_m = filter_unique_depths(d, r, ids)
                for id in jnp.unique(v_m):
                    mask = v_m == id
                    size = mask.sum()
                    if size > MAX_SIZE:
                        continue

                    o_d, o_r = v_d[:, mask], v_r[:, mask]

                    p_d, p_r, p_m = make_centered_padded_data(o_d, o_r)
                    label = True if id == 0 else False

                    save_i = save_data(p_d, p_r, p_m, label, save_i)
                file_path.unlink()
            if i % 20 == 0:
                f.flush()
