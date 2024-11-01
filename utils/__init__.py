import re
import json
import torch
import imageio
import hashlib
import trimesh
import numpy as np

import point_cloud_utils as pcu

from .generate_obj_pic import generate_obj_pics
from tqdm import tqdm
from pathlib import Path

import random

import sys
sys.path.append('..')
from eval.visualize import visualize_obj_high_q

def smooth_mesh(mesh: trimesh.Trimesh):
    mesh.export("temp-smooth.ply")
    v, f = pcu.load_mesh_vf("temp-smooth.ply")
    v_smooth = pcu.laplacian_smooth_mesh(v, f, num_iters=4, use_cotan_weights=True)
    pcu.save_mesh_vf("temp-smooth.ply", v_smooth, f)
    return trimesh.load("temp-smooth.ply")


def fit_into_bounding_box(points_sdf, bbx):
    points = points_sdf[:, 0:3]
    sdfs = points_sdf[:, [3]]

    points = points.cpu().numpy()
    sdfs = sdfs.cpu().numpy()

    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    center, lxyz = bbx
    center, lxyz = np.array(center), np.array(lxyz)
    tg_min_bound = center - (lxyz / 2)
    tg_max_bound = center + (lxyz / 2)

    # tg_min_bound, tg_max_bound = np.array(bbx[0]), np.array(bbx[1])

    max_bound[(max_bound - min_bound) < 1e-5] += 0.0001
    tg_max_bound[(tg_max_bound - tg_min_bound) < 1e-5] += 0.0001

    points = tg_min_bound + (tg_max_bound - tg_min_bound) * (
        (points - min_bound) / (max_bound - min_bound)
    )

    new_points_sdf = np.concatenate((points, sdfs), axis=1)

    cube_0 = (max_bound - min_bound).prod()     # 1
    cube_1 = (tg_max_bound - tg_min_bound).prod() # 2

    return new_points_sdf, cube_0 / cube_1

def generate_random_string(length):
    characters = 'abcdefghijklmnopqrstuvwxyz' + '0123456789'
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def str2hash(ss):
    return int(hashlib.md5(ss.encode()).hexdigest(), 16)

def camel_to_snake(name):
    # StorageFurniture -> storage furniture
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()

def generate_gif_toy(tokens, shape_output_path: Path, bar_prompt:str='', n_frame: int=100,
                    n_timepoint: int=50, fps:int=40, blender_generated_gif=False):

    def speed_control_curve(n_frame, n_timepoint, timepoint):
        frameid = n_frame*(1.0/(1+np.exp(
                -0.19*(timepoint-n_timepoint/2)))-0.5)+(n_frame/2)
        if frameid < 0:
            frameid = 0
        if frameid >= n_frame:
            frameid = n_frame-1
        return frameid

    buffers = []
    for ratio in tqdm(np.linspace(0, 1, n_frame), desc=bar_prompt):
        if not blender_generated_gif:
            buffer = generate_obj_pics(tokens, ratio,
                    [(3.487083128152961, 1.8127192062148014, 1.9810015800028038),
                    (-0.04570716149497277, -0.06563260832821388, -0.06195879116203942),
                    (-0.37480300238091124, 0.9080915656577206, -0.18679512249404312)])
        else:
            from PIL import Image
            visualize_obj_high_q(tokens, shape_output_path / "temp#1" / str(ratio), shape_output_path / "temp#2" / str(ratio), ratio)
            image = Image.open(shape_output_path / "temp#2" / str(ratio) / "result.png")
            buffer = np.array(image)
        buffers.append(buffer)

    frames = []
    for timepoint in range(n_timepoint):
        buffer_id = speed_control_curve(n_frame, n_timepoint, timepoint)
        frames.append(buffers[int(buffer_id)])

    frames = frames + frames[::-1]

    imageio.mimsave((shape_output_path / "result.gif").as_posix(), frames, fps=fps)

def untokenize_part_info(token):
    part_info = {
        'bbx': [
            token[3:6],
            token[0:3]
        ],
        'joint_data_origin': token[6:9],
        'joint_data_direction': token[9:12],
        'limit': token[12:16],
        'latent_code': token[16:],
    }
    assert len(token[16:]) == 768
    return part_info

def tokenize_part_info(part_info):
    token = []

    bounding_box = part_info['bbx']
    token += bounding_box[1]    \
           + bounding_box[0]

    joint_data_origin = part_info['joint_data_origin']
    token += joint_data_origin

    joint_data_direction = part_info['joint_data_direction']
    token += joint_data_direction

    limit = part_info['limit']
    token += limit

    latent_code = part_info['latent_code']
    token += latent_code

    return token

def generate_special_tokens(dim, seed):
    np.random.seed(seed)
    token = np.random.normal(0, 1, dim).tolist()
    return token

def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.to('cuda')
    elif isinstance(obj, dict):
        return {k: to_cuda(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cuda(v) for v in obj]
    elif isinstance(obj, tuple):
        return (to_cuda(v) for v in obj)
    else:
        return obj

class HighPrecisionJsonEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, float):
            return format(obj, '.40f')
        return json.JSONEncoder.encode(self, obj)

def parse_config_from_args():
    import argparse
    import yaml
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config',
                        help=('config file.'), required=True)
    parser.add_argument("--accelerator", default='gpu', help="The accelerator to use.")
    parser.add_argument("--devices", default=1, help="The number of devices to use.")

    args = parser.parse_args()
    config = yaml.safe_load(open(parser.parse_args().config).read())
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['accelerator'] = args.accelerator
    config['devices'] = args.devices
    return config