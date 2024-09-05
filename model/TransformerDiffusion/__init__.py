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
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.adam import Adam
from utils.base import TransArticulatedBaseModule
from .transformer.decoder import TransformerDecoder
from .diffusion.diffusion import DiffusionNet
from .diffusion.diffusion_wapper import DiffusionModel
from utils.logging import Log

from model.SDFAutoEncoder import SDFAutoEncoder

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
        self.diff_config = config['diffusion_model_paramerter']
        self.part_structure = config['part_structure']

        self.transformer = TransformerDecoder(config)
        diffusion_core = DiffusionNet(**self.diff_config['diffusion_model_config'])
        self.diffusion = DiffusionModel(diffusion_core, config)

        self.e_config = config['evaluation']
        self.sdf = SDFAutoEncoder.load_from_checkpoint(self.e_config['sdf_model_path'])
        self.sdf.eval()
        self.e_config['eval_mesh_output_path'] = Path(self.e_config['eval_mesh_output_path'] )
        self.e_config['eval_mesh_output_path'].mkdir(parents=True, exist_ok=True)

        if 'logger' in config:
            self.w_logger = config['logger']

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
            { 'params': self.transformer.parameters(), 'lr':self.op_config['tf_lr'] },
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

        # import pdb; pdb.set_trace();

        pr_token_con, vq_loss = self.transformer(input, padding_mask, enc_data)

        pr_non_pad_token_con = pr_token_con[(padding_mask > 0.5)]
        gt_non_pad_token = output[(padding_mask > 0.5)]

        pr_condition = pr_non_pad_token_con[:, -dim_condition:]
        gt_latent = gt_non_pad_token[:, -dim_latent:]

        tf_loss = F.mse_loss(pr_non_pad_token_con[:, :-dim_condition], gt_non_pad_token[:, :-dim_latent], reduction='mean')

        diff_loss_1, diff_100_loss_1, diff_1000_loss_1, pred_latent_1, perturbed_pc_1 =   \
            self.diffusion.diffusion_model_from_latent(gt_latent, cond=pr_condition)
        # diff_loss = F.mse_loss(gt_latent, pr_condition, reduction='mean')

        loss = tf_loss + diff_loss_1 + vq_loss

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
        input, output, padding_mask, end_token_mask, enc_data, _ = batch
        '''
            padding_mask:    1 -> not padding token, 0 -> padding token
            end_token_mask:  1 -> not end token,     0 -> end token
        '''
        dim_condition = self.part_structure['condition']
        dim_latent = self.part_structure['latentcode']

        pr_token_con, vq_loss = self.transformer(input, padding_mask, enc_data)

        # import pdb; pdb.set_trace();

        pr_non_pad_token_con = pr_token_con[(padding_mask > 0.5)]
        gt_non_pad_token = output[(padding_mask > 0.5)]
        end_token_non_pad_mask = end_token_mask[(padding_mask > 0.5)]

        pr_condition = pr_non_pad_token_con[:, -dim_condition:]
        gt_latent = gt_non_pad_token[:, -dim_latent:]

        # z = pr_condition
        diff_loss, diff_100_loss, diff_1000_loss, pred_z, _ =   \
            self.diffusion.diffusion_model_from_latent(gt_latent, cond=pr_condition)

        pred_z = pred_z.view_as(gt_latent)
        pred_z = pred_z[(end_token_non_pad_mask > 0.5)]
        gt_z = gt_latent[(end_token_non_pad_mask > 0.5)]

        if batch_idx == 0:
            images = []
            for z in [pred_z, gt_z]:
                # batched_recon_latent = return_dict["reconstructed_plane_feature"]
                batched_recon_latent = self.sdf.vae_model.decode(z) # reconstruced triplane features
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
            self.logger.log_image(key="Image", images=[wandb.Image(images)])