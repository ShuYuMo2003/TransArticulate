import torch
from pathlib import Path
from tqdm import tqdm

from ..mylogging import Log
from .gensdf_generator import Generator3DSDF


class GenSDFLatentCodeEvaluator:
    def __init__(self, sdf_model: torch.nn.Module, eval_mesh_output_path: Path,
                 resolution: int, max_batch: int, device: str):
        self.device = device
        self.gensdf = sdf_model
        self.gensdf.eval()

        self.eval_mesh_output_path = eval_mesh_output_path
        eval_mesh_output_path.mkdir(exist_ok=True, parents=True)

        self.generator = Generator3DSDF(
            model=self.gensdf,
            threshold=0,
            points_batch_size = max_batch,
            resolution0=resolution,
            upsampling_steps=4,
            # refinement_step=30, # Same Error as @ https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795
            device=device,
            positive_inside=False,
            simplify_nfaces=10000
        )

        Log.info('Resolution = %d', resolution)
        Log.info('Max Batch = %d', max_batch)

    def generate_mesh(self, z: torch.Tensor):
        '''
            z: latent code with 1 * dim_z
        '''
        recon_latents = self.gensdf.vae_model.decode(z)
        return self.generator.generate_from_latent(recon_latents[[0]])
        pass

    def screenshoot(self, z: torch.Tensor, mask: torch.Tensor, cut_off: int):
        z = z.reshape(-1, z.size(-1))
        mask = mask.reshape(-1)
        z = z[mask][:cut_off]

        # Comparing with OnetLatentCodeEvaluator,
        # an additional VEA model is inserted between the pointnet encoder / sdf decoder.
        # `z` represent the latent code between the vae model,
        # and `recon_latent` represent the vector that should be passed to
        # the sdf model as well as the vector generated by vae decoder.
        recon_latents = self.gensdf.vae_model.decode(z)

        images = []

        for batch in tqdm(range(z.shape[0]),
                          desc="Generating Mesh"):
            recon_latent = recon_latents[[batch]]
            '''
                The method of `SDFModel to Mesh` used by GenSDF is slower and loww accurate comparing
                with the method used by ONet.
            '''
            # output_mesh = (self.eval_mesh_output_path / f'GenSDFLatentCodeEvaluator_{batch}.ply').as_posix()
            # MeshUtils.create_mesh(self.gensdf.sdf_model, recon_latent,
            #                 output_mesh, N=self.resolution,
            #                 max_batch=self.max_batch,
            #                 from_plane_features=True)
            # mesh = trimesh.load(output_mesh)

            mesh = self.generator.generate_from_latent(recon_latent)
            mesh.export((self.eval_mesh_output_path / f'GenSDFLatentCodeEvaluator_{batch}.ply').as_posix())

            screenshot = generate_mesh_screenshot(mesh)
            images.append(screenshot)

        return images