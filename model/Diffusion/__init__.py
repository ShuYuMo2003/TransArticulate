import wandb
import torch
import trimesh
import numpy as np
import lightning as L
from pathlib import Path
import utils.mesh as MeshUtils
from torch import nn
from tqdm import tqdm
from utils.base import TransArticulatedBaseModule
from model.SDFAutoEncoder import SDFAutoEncoder

from einops.layers.torch import Rearrange

from .diffusion import DiffusionNet
from .diffusion_wapper import DiffusionModel
from .utils.helpers import ResnetBlockFC
from .vq_embedding import VQEmbedding

from utils.logging import Log


class Diffusion(TransArticulatedBaseModule):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.diff_config = config['diffusion_model_paramerter']

        diffusion_core = DiffusionNet(**self.diff_config['diffusion_model_config'])
        self.model = DiffusionModel(diffusion_core, config)
        self.compress_latentcode = nn.Sequential(*([
            ResnetBlockFC(self.diff_config['dim_latentcode'])
            for _ in range(self.diff_config['z_compress_depth'])
        ] + [
            nn.Linear(self.diff_config['dim_latentcode'],
                      self.diff_config['diffusion_model_config']['z_hat_dim']), # bottleneck
        ]))

        self.t_config = self.diff_config['text_condition']

        seqXdmodel = self.t_config['padding_length'] * self.t_config['d_model']
        seqXcomdmodel = self.t_config['padding_length'] * self.t_config['compressed_d_model']

        self.compress_text_conditon_a = nn.Sequential(*([
                nn.Linear(self.t_config['d_model'], self.t_config['compressed_d_model']),
                Rearrange('b s d -> b (s d)'),
            ]+[
                ResnetBlockFC(seqXcomdmodel) for _ in range(self.t_config['resnet_deepth'])
            ]+[
                nn.Linear(seqXcomdmodel, self.t_config['vq_width'] * self.t_config['vq_height'] * self.t_config['vq_dim_emb']),
                Rearrange('b (h w c) -> b c w h', c=self.t_config['vq_dim_emb'], w=self.t_config['vq_width'], h=self.t_config['vq_height']),
        ]))
        self.compress_text_conditon_b = VQEmbedding(n_e=self.t_config['vq_n_emb'], e_dim=self.t_config['vq_dim_emb'], beta=self.t_config['vq_beta'])
        hwc_total = self.t_config['vq_width'] * self.t_config['vq_height'] * self.t_config['vq_dim_emb']
        self.compress_text_conditon_c = nn.Sequential(*([
                Rearrange('b c w h -> b (h w c)', c=self.t_config['vq_dim_emb'], w=self.t_config['vq_width'], h=self.t_config['vq_height']),
            ]+[
                ResnetBlockFC(hwc_total) for _ in range(self.t_config['resnet_deepth'])
            ]+[
                nn.Linear(hwc_total, self.diff_config['diffusion_model_config']['text_hat_dim']),
        ]))

        self.e_config = config['evaluation']
        self.e_config['eval_mesh_output_path'] = Path(self.e_config['eval_mesh_output_path'] )
        self.e_config['eval_mesh_output_path'].mkdir(parents=True, exist_ok=True)
        self.sdf = SDFAutoEncoder.load_from_checkpoint(self.e_config['sdf_model_path'])
        self.sdf.eval()

    def configure_optimizers(self):
        return torch.optim.Adam(list(self.model.parameters()) +
                                list(self.compress_latentcode.parameters()) +
                                list(self.compress_text_conditon_a.parameters()) +
                                list(self.compress_text_conditon_b.parameters()) +
                                list(self.compress_text_conditon_c.parameters()),
                                lr=self.config['lr'])

    def step(self, batch, batch_idx):
        text, z, bbox_ratio = batch

        z_hat = self.compress_latentcode(z)

        # z_hat = torch.zeros(z.shape[0], self.diff_config['diffusion_model_config']['z_hat_dim'], deivce=z.device)

        text                    = self.compress_text_conditon_a(text)
        vq_loss, text, _, _, _  = self.compress_text_conditon_b(text)
        text_hat                = self.compress_text_conditon_c(text)

        # import pdb; pdb.set_trace()

        diff_loss_1, diff_100_loss_1, diff_1000_loss_1, pred_latent_1, perturbed_pc_1 =   \
            self.model.diffusion_model_from_latent(z, cond={
                'z_hat': z_hat,
                'text': text_hat,
                'bbox_ratio': bbox_ratio
            })

        loss = vq_loss + diff_loss_1

        data = {
            'z': z,
            'pred_latent_1': pred_latent_1,
            'loss': loss,
            'vq_loss': vq_loss,
            'diff_loss_1': diff_loss_1,
            'diff_100_loss_1': diff_100_loss_1,
            'diff_1000_loss_1': diff_1000_loss_1,
        }

        return data

    def training_step(self, batch, batch_idx):
        self.train()
        result = self.step(batch, batch_idx)

        del result['pred_latent_1']
        del result['z']
        self.log_dict(result)

        return result['loss']

    def validation_step(self, batch, batch_idx):
        if batch_idx != 0: return
        self.eval()

        result = self.step(batch, batch_idx)
        pred_latent_1 = result['pred_latent_1']
        z = result['z']

        images = []
        for z in [pred_latent_1, z]:
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

        try:
            self.logger.log_image(key="Image", images=[wandb.Image(images)])
        except Exception as e:
            Log.error(f"Error while logging images: {e}")


