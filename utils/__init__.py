import re
import json
import torch
import imageio
import hashlib
import numpy as np

from .generate_obj_pic import generate_obj_pics
from tqdm import tqdm
from pathlib import Path

def str2hash(ss):
    return int(hashlib.md5(ss.encode()).hexdigest(), 16)

def camel_to_snake(name):
    # StorageFurniture -> storage furniture
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()

def generate_gif_toy(tokens, shape_output_path: Path, bar_prompt:str='', n_frame: int=100,
                    n_timepoint: int=50, fps:int=40):

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
        buffer = generate_obj_pics(tokens, ratio,
                [(3.487083128152961, 1.8127192062148014, 1.9810015800028038),
                (-0.04570716149497277, -0.06563260832821388, -0.06195879116203942),
                (-0.37480300238091124, 0.9080915656577206, -0.18679512249404312)])
        buffers.append(buffer)

    frames = []
    for timepoint in range(n_timepoint):
        buffer_id = speed_control_curve(n_frame, n_timepoint, timepoint)
        frames.append(buffers[int(buffer_id)])

    frames = frames + frames[::-1]

    imageio.mimsave(shape_output_path.as_posix(), frames, fps=fps)

def untokenize_part_info(token):
    part_info = {
        'obbx': {
            'center': token[0:3],
            'R': [token[3:6], token[6:9], token[9:12]],
            'extent': token[12:15]
        },
        'joint_data_origin': token[15:18],
        'joint_data_direction': token[18:21],
        'limit': token[21:24],
        'latent_code': token[24:],
    }
    assert len(token[24:]) == 768
    return part_info

def tokenize_part_info(part_info):
    token = []

    bounding_box = part_info['obbx']
    token += bounding_box['center']     \
           + bounding_box['R'][0]       \
           + bounding_box['R'][1]       \
           + bounding_box['R'][2]       \
           + bounding_box['extent']

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