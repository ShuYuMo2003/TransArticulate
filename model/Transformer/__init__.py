import torch
import lightning as L

import torch.nn.functional as F

import numpy as np
import utils.mesh as MeshUtils
import wandb
import trimesh
import yaml

from tqdm import tqdm
from pathlib import Path
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.adam import Adam
from utils.base import TransArticulatedBaseModule
from .transformer.decoder import TransformerDecoder
from ..Diffusion.diffusion import DiffusionNet
from ..Diffusion.diffusion_wapper import DiffusionModel
from ..Diffusion.utils.helpers import ResnetBlockFC
from utils.logging import Log

from model.SDFAutoEncoder import SDFAutoEncoder
from model.Diffusion import Diffusion

def load_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)  # 使用 safe_load 以确保安全解析
    return config

class TransDiffusionCombineModel(TransArticulatedBaseModule):
    def __init__(self, config):
        super().__init__(config)

        self.automatic_optimization = False

        self._device = config['device']
        self.config = config
        self.op_config = config['optimizer_paramerter']
        self.tf_config = config['transformer_model_paramerter']
        # self.diff_config = config['diffusion_model_paramerter']
        self.part_structure = config['part_structure']

        self.transformer = TransformerDecoder(config)

        Log.info('Using pretrained diffusion model: %s', config['diffusion_model']['pretrained_model_path'])
        self.diffusion = Diffusion.load_from_checkpoint(config['diffusion_model']['pretrained_model_path'])

        self.diff_config = self.diffusion.diff_config

        self.condition_post_processor = nn.Sequential(*[
            ResnetBlockFC(self.part_structure['condition'])
            for _ in range(4)
        ])
        self.to_z_hat_fc = nn.Linear(self.part_structure['condition'], self.diff_config['diffusion_model_config']['z_hat_dim'])
        self.to_text_hat_fc = nn.Linear(self.part_structure['condition'], self.diff_config['diffusion_model_config']['text_hat_dim'])

        Log.info('Using pretrained SDF model: %s', config['evaluation']['sdf_model_path'])
        self.e_config = config['evaluation']
        self.sdf = SDFAutoEncoder.load_from_checkpoint(self.e_config['sdf_model_path'])
        self.sdf.eval()
        self.e_config['eval_mesh_output_path'] = Path(self.e_config['eval_mesh_output_path'])
        self.e_config['eval_mesh_output_path'].mkdir(parents=True, exist_ok=True)

        self.z_hat_dropout = nn.Dropout(self.config['diffusion_model']['z_hat_dropout'])


    # @from: https://nlp.seas.harvard.edu/annotated-transformer/#batches-and-masking
    @classmethod
    def rate(cls, step, model_size, factor, warmup):
        """
        we have to default the step to 1 for LambdaLR function
        to avoid zero raising to negative power.
        """
        if step == 0:
            step = 1
        return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        )

    def configure_optimizers(self):
        para_list = [
            { 'params': list(self.transformer.parameters()) +
                        list(self.condition_post_processor.parameters()) +
                        list(self.to_z_hat_fc.parameters()) +
                        list(self.to_text_hat_fc.parameters()) +
                        list(self.z_hat_dropout.parameters()), 'lr':self.op_config['tf_lr'] },
            { 'params': self.diffusion.parameters(), 'lr':self.op_config['diff_lr'] }
        ]
        optimizer = Adam(para_list, betas=self.op_config['betas'], eps=float(self.op_config['eps']))
        lr_scheduler = LambdaLR(optimizer,
                                lr_lambda=lambda step:
                                self.rate(step, self.tf_config['d_model'],
                                self.op_config['scheduler_factor'],
                                self.op_config['scheduler_warmup']))
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        self.train()
        input, output, padding_mask,   \
            end_token_mask, enc_data, enc_data_raw = batch
        '''
            padding_mask:    1 -> not padding token, 0 -> padding token
            end_token_mask:  1 -> not end token,     0 -> end token
        '''
        dim_condition = self.part_structure['condition']
        dim_latent = self.part_structure['latentcode']

        pr_token_con, vq_loss = self.transformer(input, padding_mask, enc_data)

        #################### Transformer Loss BEDIN ####################
        pr_non_pad_token_con = pr_token_con[(padding_mask > 0.5)]
        gt_non_pad_token = output[(padding_mask > 0.5)]

        pr_articulate_info = pr_non_pad_token_con[:, :-dim_condition]
        gt_articulate_info = gt_non_pad_token[:, :-dim_latent]

        # For non-pad token (include the end token), calculate the mse-loss as transformer loss, `tf_loss`.
        tf_loss = F.mse_loss(pr_articulate_info, gt_articulate_info, reduction='mean')
        #################### Transformer Loss END ####################

        #################### Diffusion Loss BEGIN ####################
        # Skip the end token and pad token for diffusion loss. For non pad/end tokens, we call it `valid token`.
        pr_valid_token_con = pr_token_con[(padding_mask > 0.5) & (end_token_mask > 0.5)]
        gt_valid_token = output[(padding_mask > 0.5) & (end_token_mask > 0.5)]

        pr_valid_condition = pr_valid_token_con[:, -dim_condition:]
        gt_valid_latent = gt_valid_token[:, -dim_latent:]

        # import pdb; pdb.set_trace()

        diff_loss_1, diff_100_loss_1, diff_1000_loss_1, pred_valid_token_latent_1, perturbed_pc_1 =   \
            self.diffusion.model.diffusion_model_from_latent(gt_valid_latent, cond={
                'z_hat': self.z_hat_dropout(self.to_z_hat_fc(pr_valid_condition)),
                'text': self.to_text_hat_fc(pr_valid_condition)
            })
        #################### Diffusion Loss END ####################

        loss_ratio = self.op_config['loss_ratio']
        loss = loss_ratio['tf_loss'] * tf_loss    \
             + loss_ratio['df_loss'] * diff_loss_1  \
             + loss_ratio['vq_loss'] * vq_loss

        self.log_dict({
            'train_loss': loss,
            'train_tf_loss': tf_loss,
            'train_vq_loss': vq_loss,
            'train_diff_loss': diff_loss_1,
            'train_diff_100_loss': diff_100_loss_1,
            'train_diff_1000_loss': diff_1000_loss_1,
            'transformer_lr': optimizer.param_groups[0]['lr'],
            'diffusion_lr': optimizer.param_groups[1]['lr'],
        })

        self.manual_backward(loss)
        optimizer.step()

        if self.trainer.is_last_batch:
            scheduler = self.lr_schedulers()
            scheduler.step()


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        input, output, padding_mask,   \
            end_token_mask, enc_data, enc_data_raw = batch
        '''
            padding_mask:    1 -> not padding token, 0 -> padding token
            end_token_mask:  1 -> not end token,     0 -> end token
        '''
        dim_condition = self.part_structure['condition']
        dim_latent = self.part_structure['latentcode']

        pr_token_con, vq_loss = self.transformer(input, padding_mask, enc_data)

        #################### Transformer Loss BEDIN ####################
        pr_non_pad_token_con = pr_token_con[(padding_mask > 0.5)]
        gt_non_pad_token = output[(padding_mask > 0.5)]

        pr_articulate_info = pr_non_pad_token_con[:, :-dim_condition]
        gt_articulate_info = gt_non_pad_token[:, :-dim_latent]

        # For non-pad token (include the end token), calculate the mse-loss as transformer loss `tf_loss`.
        tf_loss = F.mse_loss(pr_articulate_info, gt_articulate_info, reduction='mean')
        #################### Transformer Loss END ####################

        #################### Diffusion Loss BEGIN ####################
        # Skip the end token and pad token for diffusion loss. For non pad/end token, we call it `valid token`.
        pr_valid_token_con = pr_token_con[(padding_mask > 0.5) & (end_token_mask > 0.5)]
        gt_valid_token = output[(padding_mask > 0.5) & (end_token_mask > 0.5)]

        pr_valid_condition = pr_valid_token_con[:, -dim_condition:]
        gt_valid_latent = gt_valid_token[:, -dim_latent:]

        # import pdb; pdb.set_trace()

        diff_loss_1, diff_100_loss_1, diff_1000_loss_1, pred_valid_token_latent_1, perturbed_pc_1 =   \
            self.diffusion.model.diffusion_model_from_latent(gt_valid_latent, cond={
                'z_hat': self.to_z_hat_fc(pr_valid_condition),
                'text': self.to_text_hat_fc(pr_valid_condition)
            })
        #################### Diffusion Loss END ####################

        if batch_idx == 0:
            images = []
            for z in [pred_valid_token_latent_1, gt_valid_latent]:

                z_batch = self.e_config['z_batch']
                # import pdb; pdb.set_trace()
                batched_recon_latent = []
                for s in range(0, z.shape[0], z_batch):
                    slice_z = z[s:min(s+z_batch, z.shape[0])]
                    slice_batched_recon_latent = self.sdf.vae_model.decode(slice_z) # reconstruced triplane features
                    batched_recon_latent.append(slice_batched_recon_latent)
                batched_recon_latent = torch.cat(batched_recon_latent, dim=0)

                evaluation_count = min(self.e_config['count'], batched_recon_latent.shape[0], z.shape[0])

                screenshots = [np.random.randn(256, 256, 3) * 255 for _ in range(evaluation_count)]
                if self.e_config['count'] > batched_recon_latent.shape[0]:
                    Log.warning('`evaluation.count` is greater than batch size. Setting to batch size')

                for batch in tqdm(range(evaluation_count), desc=f'Generating Mesh for Epoch = {batch_idx}'):
                    recon_latent = batched_recon_latent[[batch]] # ([1, D*3, resolution, resolution])
                    output_mesh = (self.e_config['eval_mesh_output_path'] / f'mesh_{self.trainer.current_epoch}_{batch}.ply').as_posix()
                    try:
                        MeshUtils.create_mesh(self.sdf, recon_latent,
                                        output_mesh, N=self.e_config['resolution'],
                                        max_batch=self.e_config['max_batch'],
                                        from_plane_features=True)
                        mesh = trimesh.load(output_mesh)
                        screenshot = MeshUtils.generate_mesh_screenshot(mesh)
                    except Exception as e:
                        Log.error(f"Error while generating mesh: {e}")
                        if "Surface level must be within volume data range" in str(e):
                            break
                        continue
                    screenshots[batch] = screenshot
                image = np.concatenate(screenshots, axis=1)
                images.append(image)
            images = np.concatenate(images, axis=0)
            try: self.logger.log_image(key="Image", images=[wandb.Image(images)])
            except Exception as e: Log.error(f"Error while logging image: {e}")