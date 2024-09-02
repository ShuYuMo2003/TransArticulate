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
from torch.utils.data import DataLoader

sys.path.append('../..')
from model.SDFAutoEncoder.dataloader import GenSDFDataset
from model.SDFAutoEncoder import SDFAutoEncoder
from utils import (to_cuda, tokenize_part_info,
                   generate_special_tokens, HighPrecisionJsonEncoder, str2hash)
from utils.logging import Log

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_token = None
end_token = None
pad_token = None
max_count_token = 0
best_ckpt_path = None

def determine_latentcode_encoder():
    global best_ckpt_path

    # onets_ckpt_paths = glob('../checkpoints/onet/*.ptn')
    # onets_ckpt_paths.sort(key=lambda x: -float(x.split('/')[-1].split('-')[0]))

    # best_ckpt_path = onets_ckpt_paths[0]

    Log.info('Using best ckpt: %s', best_ckpt_path)
    gensdf = SDFAutoEncoder.load_from_checkpoint(best_ckpt_path)
    return gensdf

def evaluate_latent_codes(gensdf):
    dataloader = DataLoader(
            GenSDFDataset(
                    dataset_dir=Path('../datasets'), train=True,
                    samples_per_mesh=16000, pc_size=4096,
                    uniform_sample_ratio=0.3
                ),
            batch_size=28, num_workers=12, pin_memory=True, persistent_workers=True
        )

    gensdf.eval()
    gensdf = gensdf.to(device)

    path_to_latent = {}
    for batch, batched_data in tqdm(enumerate(dataloader),
                                    desc=f'Evaluating Latent Code', total=len(dataloader)):
        x = to_cuda(batched_data)

        xyz = x['xyz'] # (B, N, 3)
        gt = x['gt_sdf'] # (B, N)
        pc = x['point_cloud'] # (B, 1024, 3)

        with torch.no_grad():
            plane_features = gensdf.encoder.get_plane_features(pc)
            original_features = torch.cat(plane_features, dim=1)
            out = gensdf.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]

        z = out[2]

        for batch in range(z.shape[0]):
            latent = z[batch, ...]
            latent_numpy = latent.detach().cpu().numpy()
            path = x['filename'][batch]
            path = Path(path).stem.replace('.sdf', '') + ".ply"
            path_to_latent[path] = latent_numpy
            # Log.info(f"Latent code for {path} is {latent_numpy.shape}")

    Log.info('Latent code evaluation done. count = %s', len(path_to_latent))
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
        latent = path_to_latent.get(mesh_file_name)
        if latent is None:
            return f"[Error] Latent code not found for {mesh_file_name}"
        part_info['latent_code'] = latent.tolist()

        token = tokenize_part_info(part_info)

        new_parts_info.append({
                'token': token,
                'name': part_info['name'],
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
                inferenced_token.append({'token': end_token, 'dfn': -1, 'dfn_fa' : -1})

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
    best_ckpt_path = '../../train_root_dir/SDF/checkpoint/08-29-07PM-42-50/epoch=0802-loss=0.00377.ckpt'
    transformer_dataset_path = Path('../datasets/4_transformer_dataset')
    shutil.rmtree(transformer_dataset_path, ignore_errors=True)
    transformer_dataset_path.mkdir(exist_ok=True)

    shape_info_paths = list(map(Path, glob('../datasets/1_preprocessed_info/*.json')))
    gensdf = determine_latentcode_encoder()
    path_to_latent = evaluate_latent_codes(gensdf)

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

    with open(transformer_dataset_path / 'meta.json', 'w') as f:
        json.dump({
            'start_token': start_token,
            'end_token': end_token,
            'pad_token': pad_token,
            'max_count_token': max_count_token,
            'best_ckpt_path': best_ckpt_path.replace('../', ''),
        }, f, cls=HighPrecisionJsonEncoder, indent=2)

    Log.critical('Failed count: %s', len(failed))
    Log.critical('Failed: %s', failed)

