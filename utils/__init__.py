import json
import torch
import hashlib
import numpy as np

def str2hash(ss):
    return int(hashlib.md5(ss.encode()).hexdigest(), 16)

def tokenize_part_info(part_info):
    token = []

    bounding_box = part_info['bounding_box']
    token += bounding_box[0] + bounding_box[1]

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