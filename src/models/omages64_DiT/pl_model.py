"""

"""
import os
import torch
import numpy as np

from lightning.pytorch import callbacks, Trainer, LightningModule
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from omegaconf import DictConfig, OmegaConf

from .diffusion import create_diffusion

from xgutils import sysutil, plutil, ptutil, omgutil

class Model(LightningModule):
    def __init__(self,
                    baseDiT_cfg: DictConfig,
                    compute_FID = False,
                    infer_steps = 250,
                    lr = 1e-4,
                    init_from_ckpt = None,
                 **kwargs
                 ):
        super().__init__()
        self.__dict__.update(locals())
        
        bDiT = self.model = self.baseDiT = sysutil.instantiate_from_opt(self.baseDiT_cfg)

        self.diffusion       = create_diffusion(timestep_respacing="", noise_schedule='squaredcos_cap_v2')
        
        if init_from_ckpt is not None:
            init_from_ckpt = os.path.expanduser(init_from_ckpt)
            print("Initializing model from checkpoint", init_from_ckpt)

            self.load_state_dict(torch.load(init_from_ckpt, map_location=self.device)['state_dict'], strict=False)

        self.save_hyperparameters()
        
    def forward(self, x, t, cx, y):
        return self.baseDiT(x, cx, y, t=t)

    def forward_with_cfg(self, x, t, cx, y, cfg_scale):
        return self.baseDiT.forward_with_cfg(x, cx=cx, y=y, t=t, cfg_scale=cfg_scale)

    def training_step(self, batch, batch_idx, stage='train'):
        clean_images  = batch['img'] # geometry
        label         = batch['cate_label'] 
        cond_images   = batch.get('cond_img', None) # material (8 channels) RGB, Normal xyz, Metallic, Roughness

        t = torch.randint(0, self.diffusion.num_timesteps, (clean_images.shape[0],), device=self.device)

        model_kwargs = dict(y=label, cx=cond_images)
        loss_dict = self.diffusion.training_losses(self, x_start=clean_images, t=t, model_kwargs=model_kwargs)
        loss = loss_dict["loss"].mean()

        log_key = f'{stage}/loss' 
        self.log_dict({log_key: loss},
                      prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx, stage='val'):
        return self.training_step(batch, batch_idx, stage=stage)

    def test_step(self, batch, batch_idx, stage='val'):
        return self.validation_step(batch, batch_idx, stage=stage)

    @torch.inference_mode()
    def sample(self, batch, infer_steps='250', fp16=False, **kwargs: dict):
        cond_images = batch["cond_img"].to(device=self.device, dtype=self.dtype) if batch.get("cond_img", None) is not None else None
        class_labels = batch["cate_label"].to(device=self.device).long()

        n = len(class_labels)
        z = torch.randn(n, self.baseDiT.in_channels, self.baseDiT.input_size, self.baseDiT.input_size, device=self.device, dtype=self.dtype)
        y = class_labels

        z = torch.cat([z, z,], 0)
        y_null = y * 0 + 63
        y = torch.cat([y, y_null], 0)
        cx = torch.cat([cond_images, cond_images * 0 - 1.]) if cond_images is not None else None
        model_kwargs = dict(y = y, cx = cx, cfg_scale=4.)

        self.infer_diffusion = create_diffusion(timestep_respacing=str(infer_steps), noise_schedule='squaredcos_cap_v2')

        samples = self.infer_diffusion.p_sample_loop(
                        self.forward_with_cfg, z.shape, z, clip_denoised=True, model_kwargs=model_kwargs, progress=True, device=self.device)
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        print(samples.shape, samples.dtype, "sample shape and dtype")

        return samples

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=0)
        sched = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.999)
        return {
            'optimizer': optim,
        }   


class Visualizer(plutil.VisCallback):
    def __init__(self, mode="", infer_steps='250', **kwargs):
        self.__dict__.update(locals())
        super().__init__(**kwargs)

    def compute_batch(self, batch, input_name=""):
        sampled_output = self.pl_module.sample(batch=batch, infer_steps=self.infer_steps)
        computed = dict(batch=batch, sampled_output=sampled_output)
        return computed

    def visualize_batch(self, computed, input_name=""):
        computed = ptutil.ths2nps(computed)
        print("raw sampled_output shape", computed['sampled_output'].shape)
        sampled_output = computed['sampled_output'].transpose(0, 2, 3, 1)
        sampled_output = np.clip(sampled_output * .5 + .5, 0., 1.)
        gt_img = computed["batch"]["img"].transpose(0, 2, 3, 1)
        gt_img         = np.clip(gt_img * .5 + .5, 0., 1.)
        gt_omg = computed["batch"]["omage"].transpose(0, 2, 3, 1)
        gt_omg         = np.clip(gt_omg * .5 + .5, 0., 1.)
        imgs = dict()
        imgs['A_omg_2d_GT']        = gt_img[0,...,:3]
        imgs['B_omg_2d_sampled']   = sampled_output[0,...,:3]

        if computed["batch"].get("cond_img", None) is not None:
            img         = computed["batch"]["cond_img"].transpose(0, 2, 3, 1)
            img         = np.clip(img * .5 + .5, 0., 1.)
            imgs['C_omg_2d_condition'] = img[0,...,:3]

        if "vGeo" in self.mode:
            omg = gt_img[0]
            vomg, _, _ = omgutil.preview_omg(omg, return_rendered=True, geometry_only=True, combine_vomgs=True, decoder_kwargs=dict(render_mode='segcolor', preset_blend='xgutils/assets/preset_glb.blend'))
            imgs['A_omg_geo_3d_GT'] = vomg
            omg = sampled_output[0]
            vomg, _, _ = omgutil.preview_omg(omg, return_rendered=True, geometry_only=True, combine_vomgs=True, decoder_kwargs=dict(render_mode='segcolor', preset_blend='xgutils/assets/preset_glb.blend'))
            imgs['B_omg_geo_3d_sampled'] = vomg

        if "G2T" in self.mode or "G2M" in self.mode:
            geo = computed["batch"]["cond_img"].transpose(0, 2, 3, 1)[0] * .5 + .5
            omg = np.concatenate([geo, np.zeros((geo.shape[0], geo.shape[1], 8))], axis=-1)
            assert omg.shape[-1] == 12
            if "G2T" in self.mode:
                omg[..., 7:10] = sampled_output[0]
                omg_keys = ['position', 'occupancy', 'color']
                name = 'texture'
            if "G2M" in self.mode:
                if sampled_output[0].shape[-1] == 5: # if no normal
                    sout = np.zeros((geo.shape[0], geo.shape[1], 8))
                    sout[..., 3:] = sampled_output[0] # albedo, metal, rough
                omg[..., 4:] = sout
                omg_keys = ['position', 'occupancy', 'color', 'metal', 'rough']
                name = 'material'

            vomg, _, _ = omgutil.preview_omg( gt_omg[0], return_rendered=True, geometry_only=False, omg_keys=omg_keys, combine_vomgs=True, decoder_kwargs=dict(render_mode='segcolor', preset_blend='xgutils/assets/preset_glb.blend'))
            imgs[f'B_omg_{name}_3d_gt'] = vomg
            vomg, _, _ = omgutil.preview_omg(omg, return_rendered=True, geometry_only=False, omg_keys=omg_keys, combine_vomgs=True, decoder_kwargs=dict(render_mode='segcolor', preset_blend='xgutils/assets/preset_glb.blend'))
            imgs[f'B_omg_{name}_3d_sampled'] = vomg
            

        return imgs

if __name__ == '__main__':
    main()
