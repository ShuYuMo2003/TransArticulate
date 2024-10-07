import sys
import json
import copy
import shutil
import numpy as np
import torch
from rich import print
from glob import glob
from pathlib import Path
from tqdm import tqdm

sys.path.append('../..')
from model.Diffusion import Diffusion
from utils import (to_cuda, tokenize_part_info,
                   generate_special_tokens, HighPrecisionJsonEncoder, str2hash)
from utils.mylogging import Log

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_token = None
end_token = None
pad_token = None
max_count_token = 0
best_diffusion_ckpt_point = None

def evaluate_latent_codes():
    diffusion_model = Diffusion.load_from_checkpoint(best_diffusion_ckpt_point)
    diffusion_model = diffusion_model.to(device)

    mini_z_encoder = diffusion_model.z_mini_encoder
    mini_text_encoder = diffusion_model.text_mini_encoder

    diffusion_dataset_path = Path('../datasets/2.1_text_n_latentcode')

    path_to_latent = {}

    for npz in tqdm(diffusion_dataset_path.glob('*.npz'), desc='Processing NPZ(s)'):
        data = np.load(str(npz), allow_pickle=True)
        text = data['text'].astype(np.float32)
        latent_code = data['latent_code'].astype(np.float32)
        # bbox = data['bounding_box'].astype(np.float32)

        latent_code = torch.from_numpy(latent_code).to(device).unsqueeze(0)
        text = torch.from_numpy(text).to(device).unsqueeze(0)

        _, _, _, z_logits = mini_z_encoder(latent_code, 1)
        _, text_hat = mini_text_encoder(text)

        filename = npz.stem + '.ply'
        path_to_latent[filename] = {
            'z_logits': z_logits.squeeze(0).detach().cpu().numpy().tolist(),
            'latent': latent_code.squeeze(0).detach().cpu().numpy().tolist(),
            'text_hat': text_hat.squeeze(0).detach().cpu().numpy().tolist()
        }

    return path_to_latent

def process(shape_info_path:Path, transformer_dataset_path:Path, encoded_text_paths:list[Path], path_to_latent:dict):
    global start_token, end_token, pad_token, max_count_token

    shape_info = json.loads(shape_info_path.read_text())
    meta_data = shape_info['meta']

    new_parts_info = []

    # Tokenize
    for part_info in shape_info['part']:
        # Add the latent code
        mesh_file_name = part_info['mesh']
        packed_info = path_to_latent.get(mesh_file_name)
        if packed_info is None:
            return f"[Error] Latent code not found for {mesh_file_name}"
        part_info['latent_code'] = packed_info['latent']

        token = tokenize_part_info(part_info)

        new_parts_info.append({
                'token': token,
                'name': part_info['name'],
                'packed_info': packed_info,
                'dfn_fa': part_info['dfn_fa'],
                'dfn': part_info['dfn'],
            })

    start_token = generate_special_tokens(len(new_parts_info[-1]['token']),
                                          str2hash('This is start token') & ((1 << 10) - 1))

    end_token = generate_special_tokens(len(new_parts_info[-1]['token']),
                                        str2hash('This is end token') & ((1 << 10) - 1))

    pad_token = generate_special_tokens(len(new_parts_info[-1]['token']),
                                        str2hash('This is pad token') & ((1 << 10) - 1))

    root = None
    for part_info in new_parts_info:
        if part_info['dfn_fa'] == 0:
            root = part_info

        part_info['child'] = list(filter(lambda x: x['dfn_fa'] == part_info['dfn'], new_parts_info))
        part_info['child'].sort(key=lambda x: x['name'])

    assert root is not None

    exist_node = [{'token': start_token, 'dfn': 0, 'dfn_fa' : 0, 'child': [root], 'name': 'root'}]

    datasets = []

    while True: # end and start token
        inferenced_token = []
        for node in exist_node:
            if len(node['child']) > 0:
                inferenced_token.append(node['child'][0])
                node['child'].pop(0)
            else:
                inferenced_token.append({'token': end_token, 'dfn': -1, 'dfn_fa' : -1, 'packed_info': packed_info, 'name': 'end'})

        datasets.append((copy.deepcopy(exist_node), copy.deepcopy(inferenced_token)))

        all_end = True
        for node in inferenced_token:
            if node['dfn'] != -1:
                exist_node.append(node)
                all_end = False

        if all_end: break

    # TODO: save dataset.
    prefix_name = meta_data['catecory'] + '_' + meta_data['shape_id']

    # Fetch the encoded text path.
    encoded_text_paths = list(filter(lambda x: x.stem.startswith(prefix_name), encoded_text_paths))
    encoded_text_paths = list(map(lambda x : x.as_posix().replace('../', ''), encoded_text_paths))
    encoded_text_paths.sort()

    if len(encoded_text_paths) < 2:
        return f"[Error] Encoded text path not found for {prefix_name}"

    for idx, dataset in enumerate(datasets):
        dataset_name = prefix_name + '_' + str(idx) + '.json'

        for node in dataset[0]:
            if node.get('child') is not None: del node['child']
        for node in dataset[1]:
            if node.get('child') is not None: del node['child']

        max_count_token = max(max_count_token, len(dataset[0]))

        with open(transformer_dataset_path / dataset_name, 'w') as f:
            text = json.dumps({
                    'meta': meta_data,
                    'shape_info': shape_info['part'],
                    'exist_node': dataset[0],
                    'inferenced_token': dataset[1],
                    'description': encoded_text_paths,
                }, cls=HighPrecisionJsonEncoder, indent=2)
            f.write(text)

    return f"[Success] Processed {shape_info_path} part count = {len(datasets)}"

if __name__ == '__main__':
    best_diffusion_ckpt_point = '/root/workspace/crc61cnhri0c7384uggg/TransArticulate/train_root_dir/Diff/checkpoint/10-01-10PM-55-10/diffusion-epoch=35999-loss=0.05126.ckpt'

    transformer_dataset_path = Path('../datasets/4_transformer_dataset')
    shutil.rmtree(transformer_dataset_path, ignore_errors=True)
    transformer_dataset_path.mkdir(exist_ok=True)

    shape_info_paths = list(map(Path, glob('../datasets/1_preprocessed_info/*.json')))
    path_to_latent = evaluate_latent_codes()

    encoded_text_path = Path('../datasets/3_encoded_text_condition')
    encoded_text_paths = list(map(Path, glob((encoded_text_path / '*').as_posix())))

    failed = []

    for shape_info_path in tqdm(shape_info_paths, desc="Processing shape info"):
        status = process(shape_info_path, transformer_dataset_path, encoded_text_paths, path_to_latent)
        if "Success" not in status:
            failed.append((shape_info_path.stem, status))
            Log.error("%s: %s", shape_info_path.as_posix(), status)
        else:
            Log.info("%s: %s", shape_info_path.as_posix(), status)


    diffusion_dataset_path = Path('../datasets/2.1_text_n_latentcode')
    diffusion_dataset_meta = json.loads((diffusion_dataset_path / 'meta.json').read_text())

    with open(transformer_dataset_path / 'meta.json', 'w') as f:
        json.dump({
            'start_token': start_token,
            'end_token': end_token,
            'pad_token': pad_token,
            'max_count_token': max_count_token,
            'best_diffusion_ckpt_path': best_diffusion_ckpt_point,
            'best_sdf_ckpt_path': diffusion_dataset_meta['ckpt']
        }, f, cls=HighPrecisionJsonEncoder, indent=2)

    Log.critical('Failed count: %s', len(failed))
    Log.critical('Failed: %s', failed)

