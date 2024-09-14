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

class TransDiffusionCombineModel(TransArticulatedBaseModule):
    def __init__(self, config):
        super().__init__(config)

        self.automatic_optimization = False

        self._device = config['device']
        self.config = config
        self.op_config = config['optimizer_paramerter']
        self.tf_config = config['transformer_model_paramerter']
        self.part_structure = config['part_structure']

        Log.info('Using pretrained diffusion model: %s', config['diffusion_model']['pretrained_model_path'])
        self.diffusion = Diffusion.load_from_checkpoint(config['diffusion_model']['pretrained_model_path'])
        self.diff_config = self.diffusion.diff_config
        self.config['diff_config'] = self.diffusion.diff_config
        Log.info('Loaded diffusion model')

        self.transformer = TransformerDecoder(config)

        Log.info('Using pretrained SDF model: %s', config['evaluation']['sdf_model_path'])
        self.e_config = config['evaluation']
        self.sdf = SDFAutoEncoder.load_from_checkpoint(self.e_config['sdf_model_path'])
        self.sdf.eval()
        self.e_config['eval_mesh_output_path'] = Path(self.e_config['eval_mesh_output_path'])
        self.e_config['eval_mesh_output_path'].mkdir(parents=True, exist_ok=True)
        Log.info('Loaded SDF model')

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
            { 'params': list(self.transformer.parameters()), 'lr':self.op_config['tf_lr'] },
            { 'params': self.diffusion.parameters(), 'lr':self.op_config['diff_lr'] }
        ]
        optimizer = Adam(para_list, betas=self.op_config['betas'], eps=float(self.op_config['eps']))
        lr_scheduler = LambdaLR(optimizer,
                                lr_lambda=lambda step:
                                self.rate(step, self.tf_config['d_model'],
                                self.op_config['scheduler_factor'],
                                self.op_config['scheduler_warmup']))
        return [optimizer], [lr_scheduler]

    def step(self, batch, batch_idx):
        input, output, padding_mask,   \
            raw_end_token_mask, enc_data, enc_data_raw = batch
        '''
            padding_mask:        1 -> not padding token, 0 -> padding token
            raw_end_token_mask:  1 -> not end token,     0 -> end token
        '''
        dim_condition = self.part_structure['condition']
        dim_latent = self.part_structure['latentcode']

        pred_result, vq_loss = self.transformer(input, padding_mask, enc_data)

        # do not give a s**t on the padding token at the begining.
        end_token_mask = (raw_end_token_mask[padding_mask > 0.5] > 0.5)
        output = output[padding_mask > 0.5]

        #################### end_token loss BEGIN ####################
        end_token_logits = pred_result['is_end_token_logits']
        et_loss = F.binary_cross_entropy_with_logits(end_token_logits, end_token_mask.float(), reduction='mean')
        #################### end_token loss END ####################


        #################### Transformer Loss BEDIN ####################
        pr_non_pad_articulated_info = pred_result['articulated_info'][end_token_mask]
        gt_non_pad_articulated_info = output[:, :-dim_latent][end_token_mask]

        # For non-pad token (include the end token), calculate the mse-loss as transformer loss, `tf_loss`.
        tf_loss = F.mse_loss(pr_non_pad_articulated_info,
                             gt_non_pad_articulated_info, reduction='mean')
        #################### Transformer Loss END ####################


        #################### Diffusion Loss BEGIN ####################
        condition = pred_result['condition']

        min_bbox, max_bbox = pr_non_pad_articulated_info[:, 0:3], pr_non_pad_articulated_info[:, 3:6]
        bbox_ratio = (max_bbox - min_bbox)
        bbox_ratio = bbox_ratio / bbox_ratio.pow(2).sum(dim=1, keepdim=True).sqrt()
        # Skip the end token and pad token for diffusion loss.
        condition = {
            'text': condition['text_hat_condition'][end_token_mask],
            'z_hat': condition['z_hat_condition'][end_token_mask],
            'bbox_ratio': bbox_ratio
        }
        gt_latent = output[:, -dim_latent:][end_token_mask]
        diff_loss_1, diff_100_loss_1, diff_1000_loss_1, pred_valid_token_latent_1, perturbed_pc_1 =   \
            self.diffusion.model.diffusion_model_from_latent(gt_latent, cond=condition)
        #################### Diffusion Loss END ####################

        loss_ratio = self.op_config['loss_ratio']
        loss = loss_ratio['tf_loss'] * tf_loss          \
             + loss_ratio['df_loss'] * diff_loss_1      \
             + loss_ratio['vq_loss'] * vq_loss          \
             + loss_ratio['et_loss'] * et_loss

        data = {
            'loss': loss,
            'tf_loss': tf_loss,
            'vq_loss': vq_loss,
            'diff_loss': diff_loss_1,
            'et_loss': et_loss,
            'diff_100_loss': diff_100_loss_1,
            'diff_1000_loss': diff_1000_loss_1,
            'pred_valid_token_latent': pred_valid_token_latent_1,
            'gt_valid_latent': gt_latent
        }

        return data


    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.train()

        data = self.step(batch, batch_idx)

        data['transformer_lr'] = optimizer.param_groups[0]['lr']
        data['diffusion_lr'] = optimizer.param_groups[1]['lr']

        self.manual_backward(data['loss'])
        optimizer.step()


        del data['pred_valid_token_latent']
        del data['gt_valid_latent']

        self.log_dict(data, on_step=True, on_epoch=True, prog_bar=True)

        if self.trainer.is_last_batch:
            scheduler = self.lr_schedulers()
            scheduler.step()


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        data = self.step(batch, batch_idx)

        if batch_idx == 0:
            images = []
            for z in [data['pred_valid_token_latent'], data['gt_valid_latent']]:

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