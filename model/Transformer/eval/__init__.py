
import os
import copy
import torch
import json
import time
import random
import pickle

from pathlib import Path

import torch.utils
import torch.nn.functional as F
from tqdm import trange
# from rich import print
from transformers import AutoTokenizer, T5EncoderModel
from ..dataloader import TransDiffusionDataset
from .. import TransDiffusionCombineModel
from model.SDFAutoEncoder import SDFAutoEncoder

from utils import untokenize_part_info, generate_gif_toy, fit_into_bounding_box
from utils.por_cuda import POR
import utils.mesh as MeshUtils
from utils.mylogging import Log
from utils.z_to_mesh import GenSDFLatentCodeEvaluator

class Evaluater():
    def __init__(self, eval_config):
        self.eval_config = eval_config
        self.device = eval_config['device']
        self.number_of_trial = self.eval_config['number_of_trial']

        Log.info("Loading model %s", TransDiffusionCombineModel)
        self.model = TransDiffusionCombineModel.load_from_checkpoint(eval_config['checkpoint_path'])
        self.model.eval()
        # self.model.diffusion.model.cond_dropout = False
        self.m_config = self.model.config

        self.z_mini_encoder = self.model.z_mini_encoder

        d_configs = self.m_config['dataset_n_dataloader']

        self.dataset = TransDiffusionDataset(dataset_path=d_configs['dataset_path'],
                cut_off=d_configs['cut_off'],
                enc_data_fieldname=d_configs['enc_data_fieldname'],
                cache_data=False)

        self.eval_output_path = Path(self.eval_config['eval_output_path']) / time.strftime("%m-%d-%I%p-%M-%S")
        os.makedirs(self.eval_output_path, exist_ok=True)

        self.start_token = copy.deepcopy(self.dataset.start_token).to(self.device)
        self.end_token = copy.deepcopy(self.dataset.end_token).to(self.device)

        Log.info("Loading model %s", T5EncoderModel)
        self.tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large', cache_dir='cache/t5_cache')
        self.text_encoder = T5EncoderModel.from_pretrained('google-t5/t5-large', cache_dir='cache/t5_cache').to(self.device)
        #TODO: check need to do self.text_encoder.eval() or not
        self.text_encoder.eval()
        self.t5_max_sentence_length = self.eval_config['t5_max_sentence_length']

        # self.equal_part_threshold = self.eval_config['equal_part_threshold']

        # self.latentcode_evaluator = LatentCodeEvaluator(Path(self.dataset.get_onet_ckpt_path()), 100000, 16, self.device)

        Log.info("Loading model %s", SDFAutoEncoder)
        self.gensdf_config = self.eval_config['gensdf_latentcode_evaluator']
        self.gensdf_config['gensdf_model_path'] = self.dataset.get_best_sdf_ckpt_path()
        self.sdf = SDFAutoEncoder.load_from_checkpoint(self.gensdf_config['gensdf_model_path'])
        self.sdf.eval()
        self.latentcode_evaluator = GenSDFLatentCodeEvaluator(self.sdf, eval_mesh_output_path=self.eval_output_path,
                                                             resolution=self.gensdf_config['resolution'],
                                                             max_batch=self.gensdf_config['max_batch'],
                                                             device=self.device)

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
        length = token.size(0)
        difference = torch.nn.functional.mse_loss(token[:length], self.end_token[:length])
        Log.info('    - Difference with end token: %s', difference.item())
        return difference < self.equal_part_threshold

    def inference_from_text(self, text):
        Log.info('[1] Inference text: %s', text)
        encoded_text = self.encode_text(text)
        exist_node = {
            'fa': torch.tensor([0]).to(self.device),
            'token': copy.deepcopy(self.start_token).unsqueeze(0).to(self.device),
        }
        round = 1
        Log.info('[2] Generate nodes')

        dim_condition = self.m_config['part_structure']['condition']
        dim_latent = self.m_config['part_structure']['latentcode']

        while True:
            current_length = exist_node['token'].size(0)
            Log.info('   - Generate nodes round: %s, part count: %s', round, exist_node['token'].size(0))
            with torch.no_grad():
                # input: (batch, seq, xxx) ---> (batch|seq, xxx) base on `padding_mask`, the dimension of batch & seq are merged.
                # batch=1 for evaluation.
                output, vq_loss = self.model.transformer({
                                        'fa': exist_node['fa'].unsqueeze(0),        # batched.
                                        'token': exist_node['token'].unsqueeze(0),
                                    },
                                    self.generate_non_padding_mask(current_length),
                                    encoded_text) # unbatched.

            # Solve End Token.
            # True -> not end token, False -> end token
            end_token_mask = output['is_end_token_logits'] > 0
            Log.info('   - Check end token: %s', output['is_end_token_logits'])
            Log.info('   - Check end token mask: %s', end_token_mask)
            if not torch.any(end_token_mask):
                break

            condition = output['condition']
            pred_text_hat = condition['text_hat'][end_token_mask]
            pred_z_logits = condition['z_logits'][end_token_mask]

            q_z, _KL, _perplexity, _logits = self.z_mini_encoder.forward_with_logits_or_x(tau=0.5, logits=pred_z_logits)

            Log.info('   - Generate latent code with condition')
            latent = self.model.diffusion.model.generate_conditional({
                'z_hat': q_z,
                'text': pred_text_hat,
            })

            articulated_info = output['articulated_info'][end_token_mask]

            result = torch.cat((articulated_info, latent), dim=-1)

            fa_idx = torch.arange(end_token_mask.shape[0], device=self.device)
            fa_idx = fa_idx[end_token_mask]

            for idx, child_node in zip(fa_idx, result):
                exist_node['fa'] = torch.cat((exist_node['fa'], torch.tensor([idx]).to(self.device)), dim=0)
                exist_node['token'] = torch.cat((exist_node['token'], child_node.unsqueeze(0)), dim=0)

        processed_nodes = []

        Log.info('[3] Generate mesh')

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
            # import pdb; pdb.set_trace()
            part_info['z'] = z
            raw_points_sdf = self.latentcode_evaluator.generate_uniform_point_cloud_inside_mesh(z.unsqueeze(0))
            part_info['points'], part_info['rho'] = fit_into_bounding_box(raw_points_sdf, part_info['bbx'])

            processed_node.update(part_info)
            processed_nodes.append(processed_node)

        # import pdb; pdb.set_trace()

        # We do not want start token.
        return processed_nodes[1:]

    def inference(self, text):
        number_of_trial = self.number_of_trial
        list_processed_nodes = [None] * number_of_trial
        for trial in trange(number_of_trial, desc="Doing trial"):
            processed_nodes = self.inference_from_text(text)
            list_processed_nodes[trial] = {
                'data': processed_nodes,
                'rate': POR(processed_nodes, n_sample=8192),
            }
            rate = list_processed_nodes[trial]['rate']
            output_gif_path = (Path(self.eval_output_path) / f'output_{trial}_{rate}.gif')
            Log.info('[4] Generate Gif: %s', output_gif_path.as_posix())

            generate_gif_toy(processed_nodes, output_gif_path,
                            bar_prompt="   - Generate Frames")
            Log.info('[5] Done')

        output_json_path = (Path(self.eval_output_path) / f'output.json')
        output_json_path.write_text('{"text": "' + text + '"}')

        output_data_path = (Path(self.eval_output_path) / f'output.data')
        with open(output_data_path, 'wb') as f:
            f.write(pickle.dumps(list_processed_nodes))
        Log.info("Saved data checkpoint %s.", output_data_path.as_posix())