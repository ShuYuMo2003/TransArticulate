
import os
import copy
import torch
import json
import time

from pathlib import Path

import torch.utils
from tqdm import trange
from rich import print
from transformers import AutoTokenizer, T5EncoderModel
from ..dataloader import TransDiffusionDataset
from .. import TransDiffusionCombineModel
from model.SDFAutoEncoder import SDFAutoEncoder

from utils import untokenize_part_info, generate_gif_toy
import utils.mesh as MeshUtils

class Evaluater():
    def __init__(self, eval_config):
        self.eval_config = eval_config
        self.device = eval_config['device']

        self.model = TransDiffusionCombineModel.load_from_checkpoint(eval_config['checkpoint_path'])
        self.model.eval()
        self.m_config = self.model.config

        d_configs = self.m_config['dataset_n_dataloader']

        self.dataset = TransDiffusionDataset(dataset_path=d_configs['dataset_path'],
                description_for_each_file=d_configs['description_for_each_file'],
                cut_off=d_configs['cut_off'],
                enc_data_fieldname=d_configs['enc_data_fieldname'])

        self.eval_output_path = Path(self.eval_config['eval_output_path']) / time.strftime("%m-%d-%I%p-%M-%S")
        os.makedirs(self.eval_output_path, exist_ok=True)

        self.start_token = copy.deepcopy(self.dataset.start_token).to(self.device)
        self.end_token = copy.deepcopy(self.dataset.end_token).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large', cache_dir='cache/t5_cache')
        self.text_encoder = T5EncoderModel.from_pretrained('google-t5/t5-large', cache_dir='cache/t5_cache').to(self.device)
        #TODO: check need to do self.text_encoder.eval() or not
        self.text_encoder.eval()
        self.t5_max_sentence_length = self.eval_config['t5_max_sentence_length']

        self.equal_part_threshold = self.eval_config['equal_part_threshold']

        # self.latentcode_evaluator = LatentCodeEvaluator(Path(self.dataset.get_onet_ckpt_path()), 100000, 16, self.device)

        e_config = self.eval_config['gensdf_latentcode_evaluator']
        self.e_config['sdf_model_path'] = self.dataset.get_onet_ckpt_path()
        self.sdf = SDFAutoEncoder.load_from_checkpoint(self.e_config['sdf_model_path'])
        self.sdf.eval()

    def encode_text(self, text):

        input_ids = self.tokenizer([text], return_tensors="pt", padding='max_length',
                                    max_length=self.t5_max_sentence_length).input_ids
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(input_ids)
        encoded_text = outputs.last_hidden_state.detach()
        return encoded_text

    def generate_non_padding_mask(self, len):
        return torch.ones(1, len).to(self.device)

    def is_end_token(self, token):
        difference = torch.nn.functional.mse_loss(token, self.end_token)
        return difference < self.equal_part_threshold

    def inference(self, text, output_round):
        print('[1] Inference text: ', text)
        encoded_text = self.encode_text(text)
        exist_node = {
            'fa': torch.tensor([0]).to(self.device),
            'token': copy.deepcopy(self.start_token).unsqueeze(0).to(self.device),
        }
        all_end = False
        round = 1
        print('[2] Generate nodes')
        dim_condition = self.m_config['part_structure']['condition']
        dim_latent = self.m_config['part_structure']['latentcode']
        while not all_end:
            current_length = exist_node['token'].size(0)
            print('   - Generate nodes round:', round, ', part count:', exist_node['token'].size(0))
            with torch.no_grad():
                vq_loss, output = self.model.transformer({
                                        'fa': exist_node['fa'].unsqueeze(0),        # batched.
                                        'token': exist_node['token'].unsqueeze(0),
                                    },
                                    self.generate_non_padding_mask(current_length).unsqueeze(0),
                                    encoded_text) # unbatched.

                output = output[0]

            all_end = True
            condition = output[:, -dim_condition:]
            latent = self.model.diffusion.generate_conditional(condition.shape[0], condition)
            output = torch.cat((output[:, :-dim_condition], latent), dim=-1)
            for idx, child_node in enumerate(output):
                if self.is_end_token(child_node):
                    continue
                exist_node['fa'] = torch.cat((exist_node['fa'], torch.tensor([idx]).to(self.device)), dim=0)
                exist_node['token'] = torch.cat((exist_node['token'], child_node.unsqueeze(0)), dim=0)
                all_end = False

        processed_nodes = []

        print('[3] Generate mesh')

        for idx in trange(exist_node['fa'].shape[0], desc='   - Generate mesh'):
            dfn_fa = exist_node['fa'][idx].item()
            token  = exist_node['token'][idx].cpu().tolist()
            processed_node = {
                'dfn': idx,
                'dfn_fa': dfn_fa,
            }
            part_info = untokenize_part_info(token)

            z = torch.tensor(part_info['latent_code']).to(self.device)
            part_info['mesh'] = self.latentcode_evaluator.generate_mesh(z.unsqueeze(0))

            processed_node.update(part_info)
            processed_nodes.append(processed_node)

        output_path = (Path(self.eval_output_path) / f'output-{output_round}.gif')
        print('[4] Generate Gif: ', output_path.as_posix())
        generate_gif_toy(processed_nodes[1:], output_path,
                         bar_prompt="   - Generate Frames")
        print('[5] Done')

    def reconstruct(self, text, file_name):
        json_path = Path(self.eval_output_path) / '1_info'
        ply_path = Path(self.eval_output_path) / '2_mesh'
        os.makedirs(json_path, exist_ok=True)
        os.makedirs(ply_path, exist_ok=True)

        print('[1] Inference text: ', text)
        encoded_text = self.encode_text(text)
        exist_node = {
            'fa': torch.tensor([0]).to(self.device),
            'token': copy.deepcopy(self.start_token).unsqueeze(0).to(self.device),
        }
        all_end = False
        round = 1
        print('[2] Generate nodes')
        while not all_end:
            current_length = exist_node['token'].size(0)
            print('   - Generate nodes round:', round, ', part count:', exist_node['token'].size(0))
            with torch.no_grad():
                output = self.model({
                                        'fa': exist_node['fa'].unsqueeze(0),        # batched.
                                        'token': exist_node['token'].unsqueeze(0),
                                    },
                                    self.generate_non_padding_mask(current_length).unsqueeze(0),
                                    encoded_text)[0] # unbatched.

            all_end = True
            for idx, child_node in enumerate(output):
                if self.is_end_token(child_node):
                    continue
                exist_node['fa'] = torch.cat((exist_node['fa'], torch.tensor([idx]).to(self.device)), dim=0)
                exist_node['token'] = torch.cat((exist_node['token'], child_node.unsqueeze(0)), dim=0)
                all_end = False

        processed_nodes = []

        print('[3] Generate mesh')
        for idx in trange(exist_node['fa'].shape[0], desc='   - Generate mesh'):
            dfn_fa = exist_node['fa'][idx].item()
            token  = exist_node['token'][idx].cpu().tolist()
            processed_node = {
                'dfn': idx + 1,
                'dfn_fa': dfn_fa + 1,
            }
            if processed_node['dfn'] == 1:
                processed_node['dfn_fa'] = 0
            part_info = untokenize_part_info(token)

            z = torch.tensor(part_info['latent_code']).to(self.device)
            # drop part_info['latent_code'] from part_info
            part_info.pop('latent_code')

            part_info['mesh'] = f'{file_name}_{idx}.ply'
            gen_mesh = self.latentcode_evaluator.generate_mesh(z.unsqueeze(0))
            gen_mesh.export(f'{ply_path}/{part_info["mesh"]}')

            processed_node.update(part_info)
            processed_nodes.append(processed_node)

        json_path = Path(json_path) / f'{file_name}.json'
        print('[4] Save json: ', json_path)
        json_file = {"meta": {}, "part": processed_nodes}
        with open(json_path, 'w') as f:
            json.dump(json_file, f, indent=4)

        print('[5] Done')

