"""

"""
import os
import torch
import numpy as np
import contextlib

from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import callbacks, Trainer, LightningModule


from imagen_pytorch.imagen_pytorch import Unet, NullUnet, Imagen
#import ema_pytorch
import torch_ema

from xgutils import sysutil, plutil, ptutil, omgutil

class Model(LightningModule):
    def __init__(self,
                    omage_channels=12,
                    num_classes=63,
                    dim = 256,
                    cond_omage_channels=0,
                    label_embed_dim = 768,
                    timesteps = 1000,
                    use_ema = True,
                    ema_decay = 0.995,
                    unet_lr = 1e-4,
                    unet_eps = 1e-8,
                    beta1 = 0.9,
                    beta2 = 0.99,
                    max_grad_norm = None,
                    group_wd_params = True,
                    warmup_steps = None,
                    cosine_decay_max_steps = None,
                    only_train_unet_number = None,
                    infer_steps=250,
                    init_from_ckpt=None,
                 **kwargs
                 ):
        super().__init__()
        self.__dict__.update(locals())
        

        self.unets = unet = Unet( # follows original paper
            channels=omage_channels,
            cond_images_channels=cond_omage_channels,
            dim = dim,
            text_embed_dim = label_embed_dim,
            dim_mults = (1, 2, 3, 4),
            num_resnet_blocks = 3,
            layer_attns = (False, True, True, True),
            layer_cross_attns = (False, True, True, True),
            attn_heads = 8,
            memory_efficient = False,
            init_conv_to_final_conv_residual = True, # try to solve color shifting problem. https://github.com/lucidrains/imagen-pytorch/issues/72#issuecomment-1272591508
            resize_mode='nearest',
        )

        # imagen, which contains the unets above (base unet and super resoluting ones)

        self.imagen = imagen = Imagen(
            channels=omage_channels,
            unets = self.unets,
            text_encoder_name = 'google/t5-v1_1-base',
            image_sizes = 64,
            timesteps = timesteps,
            cond_drop_prob = 0.1,
            pred_objectives='v', # try to solve color shifting problem. https://github.com/lucidrains/imagen-pytorch/issues/72#issuecomment-1272591508
            auto_normalize_img=False,
        )
        self.model = imagen
        self.ema = torch_ema.ExponentialMovingAverage( self.model.parameters(), decay=0.999)

        self.label_embedder = torch.nn.Embedding(num_classes, label_embed_dim)


        if init_from_ckpt is not None:
            print("Initializing model from checkpoint", init_from_ckpt)
            self.load_state_dict(torch.load(init_from_ckpt)['state_dict'], strict=False)

        self.save_hyperparameters()
    def label2embeds(self, label):
        slabel = label[:, None]
        text_embeds = self.label_embedder( slabel )
        text_masks = (slabel * 0 + 1).bool()
        return text_embeds, text_masks
    def forward(self, x, t, cx, y):
        pass

    def training_step(self, batch, batch_idx, stage='train'):
        clean_images  = batch['img'] # geometry
        label         = batch['cate_label'] 
        cond_images   = batch.get('cond_img', None) # material (8 channels) RGB, Normal xyz, Metallic, Roughness

        if cond_images is not None:
            assert cond_images.shape[1] == self.cond_omage_channels, f"cond_images.shape[1] {cond_images.shape[1]} != self.cond_omage_channels {self.cond_omage_channels}"

        text_embeds, text_masks = self.label2embeds(label)

        loss = self.imagen(images=clean_images, unet=self.unets, text_embeds=text_embeds, text_masks=text_masks, unet_number=None, cond_images=cond_images)


        log_key = f'{stage}/loss' 
        self.log_dict({log_key: loss},
                      prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx, stage='val'):
        return self.training_step(batch, batch_idx, stage=stage)

    def test_step(self, batch, batch_idx, stage='val'):
        return self.validation_step(batch, batch_idx, stage=stage)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.use_ema: 
            checkpoint['ema'] = self.ema.state_dict()
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.use_ema:
            self.ema.load_state_dict(checkpoint['ema'])
        return super().on_load_checkpoint(checkpoint)

    def on_before_zero_grad(self, optimizer) -> None:
        if self.use_ema:
            self.ema.update(self.model.parameters())
        return super().on_before_zero_grad(optimizer)
    def to(self, *args, **kwargs):
        if self.use_ema:
            self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    @contextlib.contextmanager
    def maybe_ema(self):
        ema = self.ema  # The EMACallback() ensures this
        ctx = nullcontext if ema is None else ema.average_parameters
        yield ctx

    @torch.no_grad()
    def sample(self, batch, infer_steps='250', fp16=False, **kwargs: dict):
        cond_images = batch["cond_img"].to(device=self.device, dtype=self.dtype) if batch.get("cond_img", None) is not None else None
        labels = batch["cate_label"].to(device=self.device).long()
        batch_size = len(labels)
        text_embeds, text_masks = self.label2embeds(labels)
        with self.maybe_ema():
            omages = self.imagen.sample(text_embeds=text_embeds, text_masks=text_masks, cond_images = cond_images, cond_scale=1., batch_size = batch_size, return_pil_images = False)
            
        return omages

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.unets.parameters(),
            lr = self.unet_lr,
            eps = self.unet_eps,
            betas = (self.beta1, self.beta2)
        )
        return {'optimizer': optimizer}

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
