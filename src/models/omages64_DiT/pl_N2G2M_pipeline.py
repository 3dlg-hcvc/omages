"""

"""
import os
import torch
import numpy as np

from lightning.pytorch import callbacks, Trainer, LightningModule
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from omegaconf import DictConfig, OmegaConf

from .diffusion import create_diffusion

from xgutils import sysutil, plutil, ptutil, omgutil, optutil

class Model(LightningModule):
    def __init__(self,
                 **kwargs
                 ):
        super().__init__()
        self.__dict__.update(locals())
        self.dummy_param = torch.nn.Parameter(torch.zeros(2), requires_grad=True) # dummy parameter to avoid 'RuntimeError: DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient' error.
        
    def training_step(self, batch, batch_idx, stage='train'):
        # raise NotImplementedError("This pipeline is not intended for training/testing/validation.")
        loss = self.dummy_param.sum() # dummy loss
        log_key = f'{stage}/loss'
        self.log_dict({log_key: loss},
                      prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, stage='val'):
        return self.training_step(batch, batch_idx, stage=stage)

    def test_step(self, batch, batch_idx, stage='val'):
        return self.validation_step(batch, batch_idx, stage=stage)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=1e-4, weight_decay=0)
        sched = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.999)
        return {
            'optimizer': optim,
        }

def cfg2model(path, ckpt_path):
    
    opt = optutil.get_opt(path, root_dir=os.getcwd(), src_name='src')
    cfg = OmegaConf.create(opt)
    model = sysutil.instantiate_from_opt(cfg.pl_model_opt)
    model.load_state_dict(torch.load(ckpt_path, map_location=model.device)['state_dict'])
    return model

class N2G2MCallback(plutil.ComputeGraphCallback):
    def __init__(self, N2G_cfg, N2G_ckpt, G2M_cfg, G2M_ckpt, if_load_saved_compute=[True, True, False], mode="", infer_steps='250', **kwargs):
        self.__dict__.update(locals())
        super().__init__(**kwargs)
        self.define_nodes_graph()
        self.N2G_module = cfg2model(N2G_cfg, N2G_ckpt)
        self.G2M_module = cfg2model(G2M_cfg, G2M_ckpt)
    def define_nodes_graph(self):
        ifl = self.if_load_saved_compute
        self.nodes = [  dict(type='normal', name='N2G_compute', fn=self.N2G_compute, load=ifl[0]),
                        dict(type='normal', name='G2M_compute', fn=self.G2M_compute, load=ifl[1]),
                        dict(type='image',  name='G2M_visual',  fn=self.G2M_visual,  load=ifl[2]),
                    ]
    def N2G_compute(self, data_dict, input_name=""):
        batch = data_dict
        N2G_module = self.N2G_module.to(self.pl_module.device)
        sampled_output = N2G_module.sample(batch=batch, infer_steps=self.infer_steps)
        computed = dict(batch=batch, sampled_output=sampled_output)
        return computed

    def G2M_compute(self, data_dict, input_name=""):
        batch = data_dict["batch"]
        batch['cond_img'] = data_dict['sampled_output']
        G2M_module = self.G2M_module.to(self.pl_module.device)
        sampled_output = G2M_module.sample(batch=batch, infer_steps=self.infer_steps)
        computed = dict(batch=batch, sampled_output=sampled_output)
        return computed
    def G2M_visual(self, data_dict, input_name=""):
        imgs = visualize(data_dict, input_name=input_name, mode='G2M')
        return imgs
    

def visualize( computed, input_name="", mode=''):
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

    if "vGeo" in mode:
        omg = gt_img[0]
        vomg, _, _ = omgutil.preview_omg(omg, return_rendered=True, geometry_only=True, combine_vomgs=True, decoder_kwargs=dict(render_mode='segcolor', preset_blend='xgutils/assets/preset_glb.blend'))
        imgs['A_omg_geo_3d_GT'] = vomg
        omg = sampled_output[0]
        vomg, _, _ = omgutil.preview_omg(omg, return_rendered=True, geometry_only=True, combine_vomgs=True, decoder_kwargs=dict(render_mode='segcolor', preset_blend='xgutils/assets/preset_glb.blend'))
        imgs['B_omg_geo_3d_sampled'] = vomg

    if "G2T" in mode or "G2M" in mode:
        geo = computed["batch"]["cond_img"].transpose(0, 2, 3, 1)[0] * .5 + .5
        omg = np.concatenate([geo, np.zeros((geo.shape[0], geo.shape[1], 8))], axis=-1)
        assert omg.shape[-1] == 12
        if "G2T" in mode:
            omg[..., 7:10] = sampled_output[0]
            omg_keys = ['position', 'occupancy', 'color']
            name = 'texture'
        if "G2M" in mode:
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
