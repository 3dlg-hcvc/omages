import os
import sys
import glob
import torch
import traceback

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from xgutils import sysutil, nputil, ptutil
from xgutils.vis import visutil

from abc import ABC, abstractmethod
from lightning import Callback, LightningModule, Trainer
# Pytorch Lightning
#from pytorch_lightning import Callback, LightningModule, Trainer
def dataset_generator(pl_module, dset, data_indices=[0,1,2], **get_kwargs):
    is_training = pl_module.training
    with torch.no_grad():
        pl_module.eval()
        for ind in data_indices:
            dataitem = dset.__getitem__(ind,**get_kwargs)
            batch = {}
            for key in dataitem:
                datakey = dataitem[key]
                if type(datakey) is not np.ndarray and type(datakey) is not torch.Tensor:
                    continue
                datakey = dataitem[key][None,...]
                if type(datakey) is np.ndarray:
                    datakey = torch.from_numpy(datakey)
                batch[key] = datakey.to(pl_module.device)
            yield batch
        if is_training:
            pl_module.train()
        else:
            pl_module.eval()


class FlyObj():
    """ FlyObj is used for on-the-fly data processing and saving
        If you have a sequence of type A data, and you want to first process it to type B, and then convert it to type C, and you may want to control whether all type B processing should be done before type C processing. You can control the behavior by setting on_the_fly to True or False.

    """

    def __init__(self, save_dir=None, load_dir=None, on_the_fly=True, data_processor=None):
        if data_processor is None:
            data_processor = self.dflt_data_processor
        self.__dict__.update(locals())
    def process_iter(self, input_iter):
        for name, input_data in input_iter:
            processed = self.load(name)
            if processed is None:
                processed = self.data_processor(input_data, input_name=name)
            yield name, processed

    def __call__(self, input_iter):
        process_iter = self.process_iter(input_iter)
        
        if self.on_the_fly==False:
            all_processed = list(process_iter)
            list(starmap(self.save, all_processed))
            for name, processed in all_processed:
                yield name, processed
        else:
            for name, processed in process_iter:
                self.save(name, processed)
                yield name, processed
    @staticmethod
    def dflt_data_processor(input_data):
        return input_data
    def save(self, name, data):
        if self.save_dir is not None:
            sysutil.mkdirs(self.save_dir)
            save_path = os.path.join(self.save_dir, f"{name}.npy")
            np.save(save_path, ptutil.ths2nps(data))
    def load(self, name):
        if self.load_dir is None:
            return None
        load_path = os.path.join(self.load_dir, f"{name}.npy")
        if os.path.exists(load_path) == False:
            return None
        loaded    = np.load(load_path,allow_pickle=True).item()
        return loaded
class ImageFlyObj(FlyObj):
    def save(self, name, imgs):
        if self.save_dir is not None:
            sysutil.mkdirs(self.save_dir)
            for key in imgs:
                save_path = os.path.join(self.save_dir, f"{name}_{key}.png")
                img = visutil.img2rgba(imgs[key])
                visutil.saveImg(save_path, img)
    def load(self, name):
        if self.load_dir is None:
            return None
        load_paths = os.path.join(self.load_dir, f"{name}_*.png")
        files = glob.glob(load_paths)
        files.sort(key=os.path.getmtime)
        if len(files) == 0:
            return None
        loaded = {}
        for imgf in files:
            key = "_".join(imgf[:-4].split("_")[1:])
            loaded[key] = visutil.readImg(imgf)
        return loaded
def dataset_generator(pl_module, dset, data_indices=[0,1,2], yield_ind=True, **get_kwargs):
    is_training = pl_module.training
    with torch.no_grad():
        pl_module.eval()
        for ind in data_indices:
            dataitem = dset.__getitem__(ind,**get_kwargs)
            batch = {}
            batch = ptutil.nps2ths(dataitem, device=pl_module.device)
            for key in batch:
                if type(batch[key]) is torch.Tensor:
                    batch[key] = batch[key][None,...]
            if yield_ind==True:
                yield str(ind), batch
            else:
                yield batch
        if is_training:
            pl_module.train()
        else:
            pl_module.eval()

def get_effective_visual_indices(indices, global_rank, gpu_nums):
    """ Assign different indices for different gpus
        global_rank: which gpu is the current device
        gpu_nums: total number of gpus
        [[0 5]
         [1 6]
         [2 7]
         [3]
         [4]
        ]
    """
    indices = np.array(indices)
    total_num = len(indices)
    batch_size = nputil.ceiling_division(total_num-global_rank, gpu_nums)
    effective_ind =  global_rank + gpu_nums * np.arange(batch_size)
    effective = indices[effective_ind]
    return effective
def get_effective_visual_indices_unittest():
    gpus_nums=5
    for rg in range(20):
        for i in range(gpus_nums):
            x=get_effective_visual_indices( np.arange(rg)*2, i, gpus_nums)
            print(x)
        print("!")
class VisCallback(Callback):
    """ The template callback for visualization
        Design Motivation:
            - Visualization as callback. By utilizing this pytorch-lightning functionality, we can separate the visualization code from the model code. Making each part more readable and maintainable.
            - Seperated compute and visualization. To visualize something, you need to first compute something. Sometime the compute part is very slow, and you don't want to recompute it everytime you want to visualize it. By using FlyObj, we can save the computed data, and load it when needed. In other times, the visualization part is very slow, and you want to do all the compute first, and then visualize them. By setting on_the_fly to False, we can control this behavior.
    """
    def __init__(self,  visual_indices=[0,1,2,3,4,5], all_indices=False, force_visual_indices=False, \
                        every_n_epoch=3, no_sanity_check=False, \
                        load_compute=False, load_visual=False, \
                        data_dir = None, output_name=None, use_dloader=False, num_gpus=1, 
                        parallel_vis=False, single_vis=False, visall_after_training_end=True, end_command='',**kwargs):
        super().__init__()
        self.__dict__.update(locals())
        self.classname = self.__class__.__name__
        if self.output_name == None:
            self.output_name = self.classname
        if self.data_dir is None:
            self.data_dir = f"/studio/nnrecon/temp/{self.output_name}/"
        if all_indices is True and force_visual_indices==False:
            self.visual_indices = "all"
        print("VI", self.visual_indices)
    def process(self, pl_module, dloader, visual_indices=None, data_dir=None, visual_summary=True, \
                parallel_vis=None, load_compute=None, load_visual=None, fly_compute=True, debug=False):
        self.pl_module = pl_module
        self.dset = dset = dloader.dataset
        if load_compute is None:
            load_compute = self.load_compute
        if load_visual is None:
            load_visual = self.load_visual
        if data_dir is None:
            data_dir = self.data_dir
        if visual_indices is None:
            visual_indices = self.visual_indices
        if parallel_vis is None:
            parallel_vis = self.parallel_vis
        if visual_indices == "all":
            visual_indices = list(range(len(dset)))
        #print("is force?", self.force_visual_indices)
        if self.all_indices==False or self.force_visual_indices==True:
            visual_indices = self.visual_indices
        print("self.visual_indices", self.visual_indices)
        compute_dir = os.path.join(data_dir, "computed")
        cload_dir   = compute_dir if load_compute==True else None
        visual_dir  = os.path.join(data_dir, "visual")
        vload_dir   = visual_dir  if load_visual==True  else None
        if parallel_vis==True:
            visual_indices = get_effective_visual_indices(visual_indices, pl_module.global_rank, self.num_gpus)
            print("global_rank", self.pl_module.global_rank, "/", self.num_gpus, "effecitve indices", visual_indices)
            self.single_vis = False

        if self.use_dloader==False:
            datagen      = dataset_generator(pl_module, dset, visual_indices)
        else:
            datagen      = dloader
        computegen   = FlyObj(data_processor=self.compute_batch, save_dir=compute_dir, load_dir=cload_dir, on_the_fly=fly_compute)
        visgen       = ImageFlyObj(data_processor=self.visualize_batch, save_dir=visual_dir, load_dir=vload_dir)
        imgsgen, imgs = visgen(computegen(datagen)), []
        failed_ind = []
        for ind in sysutil.progbar(visual_indices):
            try:
                item = next(imgsgen)
                if len(item[1])>0:
                    imgs.append(item)
            except Exception as exc:
                print("Failed to visualize index", ind)
                import traceback, pdb
                # Print exception details
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("Exception occurred:", exc_type)  # Print the exception type
                traceback.print_tb(exc_traceback)  # Print the stack trace
                print(getattr(exc, 'message', repr(exc)))  # Print the exception message if available
                print('\n\n\n\n\n\n\n')
                if debug==True:
                    import pdb
                    # Enter post-mortem debug mode
                    pdb.post_mortem(exc_traceback)  # Pass the traceback to post_mortem
                    # exit after debugging
                    sys.exit(0)
                failed_ind.append(ind)
        failed_log = os.path.join(self.data_dir, f"logs/failed_ind/")
        sysutil.mkdirs(failed_log)
        np.savetxt( failed_log+f"/rank_{self.pl_module.global_rank}.txt", 
                    np.array(failed_ind))
        # split the (world_size, N) images
        if visual_summary==True and len(imgs)>0:
            # all_images = imgs
            summary_imgs = self.get_summary_imgs(pl_module, imgs, zoomfac=1.)
            simg = summary_imgs[list(summary_imgs.keys())[0]]["image"]

            all_simg = pl_module.all_gather(simg) # (world_size, H, W, C)
            if self.pl_module.global_rank==0: 
                # gather all summary images from all gpus
                all_simg = [img.cpu().numpy() for img in all_simg]
                simg = visutil.imageGrid(all_simg, shape=(-1, 1), zoomfac=1)

                visutil.saveImg(os.path.join(visual_dir, "summary.png"), simg)
        else:
            summary_imgs = None

        # generate html
        if self.pl_module.global_rank == 0:
            # # wait for 300 milliseconds to make sure all images are saved
            # import time
            # time.sleep(3)
            # visual_html{epoch}
            # if current mode is training mode
            if hasattr(pl_module, "current_epoch"):
                html_dir = os.path.join(data_dir, f"visual_html_{pl_module.current_epoch}/")
            else:
                # set the name according to train/val/test
                html_dir = os.path.join(data_dir, f"visual_html")
            # show results as html and set title as experiment name
            visutil.html5_result_vis(visual_dir, html_dir, height=300)

        self.imgs, self.summary_imgs = imgs, summary_imgs
        
        if self.end_command != '' and self.pl_module.global_rank == 0:
            print("Running end command:", self.end_command)
            # import subprocess
            # subprocess.Popen(self.end_command, shell=True)
            os.system(self.end_command)

        #for l,img in visgen(computegen(datagen)):
        #    visutil.showImg(img["recon"])
        #visutil.showImg(self.summary_imgs[self.summary_imgs.keys()[0]]["image"])
        #return self.summary_imgs
    def process_all(self, pl_module, dloader, parallel_vis=True, **kwargs):
        return self.process(pl_module, dloader, parallel_vis=parallel_vis, visual_indices="all", **kwargs)
    def compute_batch(batch):
        """ example code """
        logits = batch["Ytg"].clone()
        logits[:]= torch.rand(logits.shape[0])
        return {"logits":logits, "batch":batch}
    def visualize_batch(computed):
        """ example code """
        computed = ptutil.ths2nps(computed)
        batch = computed["batch"]
        Ytg = computed["logits"]
        Ytg = batch["Ytg"]
        vert, face = geoutil.array2mesh(Ytg, thresh=.5)
        img = fresnelvis.renderMeshCloud({"vert":vert,"face":face})
        return {"recon":img}
    def on_sanity_check_end(self, trainer, pl_module):
        print(f"\n{self.__class__.__name__} callback")
        if self.single_vis==True and pl_module.global_rank!=0:
            print("single_vis==True, Only run callback on global_rank 0, current rank:", pl_module.global_rank)
            return 
        if self.no_sanity_check:
            print("no_sanity_check is set to True, skipping...")
            return
        self.process(pl_module, trainer.datamodule.visual_dataloader(), visual_summary=True)
        self.log_summary_images(trainer, pl_module, self.summary_imgs)
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs) -> None:
        if trainer.current_epoch % self.every_n_epoch == self.every_n_epoch-1:
            print(f"\n{self.__class__.__name__} callback")
            if self.single_vis==True and pl_module.global_rank!=0: 
                print("single_vis==True, Only run callback on global_rank 0, current rank:", pl_module.global_rank)
                return 
            try:
                self.process(pl_module, trainer.datamodule.visual_dataloader(), visual_summary=True)
                self.log_summary_images(trainer, pl_module, self.summary_imgs)
            except Exception as exc:
                traceback.print_tb(exc.__traceback__)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback)  # Print the stack trace
                print(getattr(exc, 'message', repr(exc)))  # Print the exception message if available
                print("Exception occurred:", exc_type)  # Print the exception type
                print("Something is wrong in the callback, skipping...")
    def on_test_start(self, trainer, pl_module, **kwargs):
        self.process_all(pl_module, trainer.datamodule.visual_dataloader(), **kwargs)
    # def on_test_end(self, trainer, pl_module, **kwargs):
    #     self.process_all(pl_module, pl_module.visual_dataloader(), **kwargs)

    def post_training_process(self, trainer, pl_module, data_module, **kwargs):
        if self.visall_after_training_end==True:
            self.process_all(pl_module, data_module.visual_dataloader(), **kwargs)
    def get_summary_imgs(self, pl_module, imgs, zoomfac=.5):
        all_images = []
        rows = len(imgs)
        for name, image_array in imgs:
            for img_name in image_array:
                img = image_array[img_name]
                all_images.append( torch.from_numpy(img) )
        # gather all images from all gpus
        # gathered = pl_module.all_gather(all_images)
        # if len(gathered[0].shape) == 4: # (world_size, H, W, C)
        #     world_size = gathered[0].shape[0]
        # else:                           # (H, W, C)
        #     world_size = 1
        #     gathered = [gimg[None,...] for gimg in gathered]
        # rows = world_size * rows
        # all_images = []
        # for i in range(world_size): # for each gpu
        #     for j in range(rows):
        #         all_images.append(gathered[j][i])
        # import pdb
        # pdb.set_trace()
        summary = visutil.imageGrid(all_images, shape=(rows, -1), zoomfac=zoomfac)
        return {self.classname: {"caption":self.classname, "image":summary}}
    def log_summary_images(self, trainer, pl_module, summary_imgs, x_axis="epoch"):
        # wandb logger
        import wandb
        for key in summary_imgs:
            t = summary_imgs[key]
            title   = key
            caption = t["caption"]
            image   = t["image"]
            #log_image(trainer, title, caption, image, trainer.global_step)
            #x_val = trainer.current_epoch if x_axis=="epoch" else trainer.global_step
            trainer.logger.experiment.log( \
                {title:[wandb.Image(image,caption=caption)], \
                    "epoch": trainer.current_epoch,
                    "global_step": trainer.global_step})


class ComputeGraphCallback(Callback):
    """ The template callback for chained computation (upgrade from VisCallback)
        Motivation:
        VisCallback is only designed for 1 step computation and visualization. 
        This class can support multiple (chained) steps of computation and visualization.
        In the future, we will add support for general Directed Acyclic Graph (DAG) computation and visualization.
    """
    def __init__(self,  visual_indices=[0,1,2,3,4,5], all_indices=False, force_visual_indices=False, \
                        every_n_epoch=3, no_sanity_check=False, \
                        load_compute=False, load_visual=False, \
                        data_dir = None, output_name=None, use_dloader=False, num_gpus=1, 
                        parallel_vis=False, single_vis=False, visall_after_training_end=True, end_command='',**kwargs):
        super().__init__()
        self.__dict__.update(locals())
        self.classname = self.__class__.__name__
        if self.output_name == None:
            self.output_name = self.classname
        if self.data_dir is None:
            self.data_dir = f"~/studio/temp/{self.output_name}/"
        if all_indices is True and force_visual_indices==False:
            self.visual_indices = "all"
        print("VI", self.visual_indices)
    def define_nodes_graph(self):
        raise NotImplementedError("Please define the nodes in the compute graph")
    def process(self, pl_module, dloader, visual_indices=None, data_dir=None, visual_summary=True, fly_compute=True, debug=False, parallel_vis=True, **kwargs):
        self.pl_module = pl_module
        self.dset = dset = dloader.dataset
        if data_dir is None:
            data_dir = self.data_dir
        if visual_indices is None:
            visual_indices = self.visual_indices
        if visual_indices == "all":
            visual_indices = list(range(len(dset)))
        #print("is force?", self.force_visual_indices)
        if self.all_indices==False or self.force_visual_indices==True:
            visual_indices = self.visual_indices
        print("self.visual_indices", self.visual_indices)
        compute_dir = os.path.join(data_dir, "computed")
        visual_dir  = os.path.join(data_dir, "visual")
        if parallel_vis==True:
            visual_indices = get_effective_visual_indices(visual_indices, pl_module.global_rank, self.num_gpus)
            print("global_rank", self.pl_module.global_rank, "/", self.num_gpus, "effecitve indices", visual_indices)
            self.single_vis = False

        if self.use_dloader==False:
            datagen      = dataset_generator(pl_module, dset, visual_indices)
        else:
            datagen      = dloader
        node_generators = []
        for i in range(len(self.nodes)):
            save_dir = os.path.join(data_dir, f"{self.nodes[i]['name']}")
            if self.nodes[i]['type'] == 'normal':
                gen = FlyObj(data_processor=self.nodes[i]['fn'], save_dir=save_dir, load_dir=(save_dir if self.nodes[i]['load']==True else None),      on_the_fly=fly_compute)
            elif self.nodes[i]['type'] == 'image':
                gen = ImageFlyObj(data_processor=self.nodes[i]['fn'], save_dir=save_dir, load_dir=(save_dir if self.nodes[i]['load']==True else None), on_the_fly=fly_compute)
            else:
                raise ValueError(f"Unknown node type: {self.nodes[i]['type']}")
            node_generators.append(gen)
        pipeline = datagen
        for gen in node_generators:
            pipeline = gen(pipeline)
        failed_ind = []

        imgs = []
        for ind in sysutil.progbar(visual_indices):
            try:
                item = next(pipeline)
                if len(item[1])>0:
                    imgs.append(item)
            except Exception as exc:
                print("Failed to visualize index", ind)
                import traceback, pdb
                # Print exception details
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("Exception occurred:", exc_type)  # Print the exception type
                traceback.print_tb(exc_traceback)  # Print the stack trace
                print(getattr(exc, 'message', repr(exc)))  # Print the exception message if available
                print('\n\n\n\n\n\n\n')
                if debug==True:
                    import pdb
                    # Enter post-mortem debug mode
                    pdb.post_mortem(exc_traceback)  # Pass the traceback to post_mortem
                    # exit after debugging
                    sys.exit(0)
                failed_ind.append(ind)
        failed_log = os.path.join(self.data_dir, f"logs/failed_ind/")
        sysutil.mkdirs(failed_log)
        np.savetxt( failed_log+f"/rank_{self.pl_module.global_rank}.txt", 
                    np.array(failed_ind))

        if self.end_command != '' and self.pl_module.global_rank == 0:
            print("Running end command:", self.end_command)
            # import subprocess
            # subprocess.Popen(self.end_command, shell=True)
            os.system(self.end_command)

        #for l,img in visgen(computegen(datagen)):
        #    visutil.showImg(img["recon"])
        #visutil.showImg(self.summary_imgs[self.summary_imgs.keys()[0]]["image"])
        #return self.summary_imgs
    def process_all(self, pl_module, dloader, parallel_vis=True, **kwargs):
        return self.process(pl_module, dloader, parallel_vis=parallel_vis, visual_indices="all", **kwargs)
    def compute_batch(batch):
        """ example code """
        logits = batch["Ytg"].clone()
        logits[:]= torch.rand(logits.shape[0])
        return {"logits":logits, "batch":batch}
    def visualize_batch(computed):
        """ example code """
        computed = ptutil.ths2nps(computed)
        batch = computed["batch"]
        Ytg = computed["logits"]
        Ytg = batch["Ytg"]
        vert, face = geoutil.array2mesh(Ytg, thresh=.5)
        img = fresnelvis.renderMeshCloud({"vert":vert,"face":face})
        return {"recon":img}
    def on_sanity_check_end(self, trainer, pl_module):
        print(f"\n{self.__class__.__name__} callback")
        if self.single_vis==True and pl_module.global_rank!=0:
            print("single_vis==True, Only run callback on global_rank 0, current rank:", pl_module.global_rank)
            return 
        if self.no_sanity_check:
            print("no_sanity_check is set to True, skipping...")
            return
        self.process(pl_module, trainer.datamodule.visual_dataloader(), visual_summary=True)
        self.log_summary_images(trainer, pl_module, self.summary_imgs)
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs) -> None:
        if trainer.current_epoch % self.every_n_epoch == self.every_n_epoch-1:
            print(f"\n{self.__class__.__name__} callback")
            if self.single_vis==True and pl_module.global_rank!=0: 
                print("single_vis==True, Only run callback on global_rank 0, current rank:", pl_module.global_rank)
                return 
            try:
                self.process(pl_module, trainer.datamodule.visual_dataloader(), visual_summary=True)
                self.log_summary_images(trainer, pl_module, self.summary_imgs)
            except Exception as exc:
                traceback.print_tb(exc.__traceback__)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback)  # Print the stack trace
                print(getattr(exc, 'message', repr(exc)))  # Print the exception message if available
                print("Exception occurred:", exc_type)  # Print the exception type
                print("Something is wrong in the callback, skipping...")
    def on_test_start(self, trainer, pl_module, **kwargs):
        self.process_all(pl_module, trainer.datamodule.visual_dataloader(), **kwargs)
    # def on_test_end(self, trainer, pl_module, **kwargs):
    #     self.process_all(pl_module, pl_module.visual_dataloader(), **kwargs)

    def post_training_process(self, trainer, pl_module, data_module, **kwargs):
        if self.visall_after_training_end==True:
            self.process_all(pl_module, data_module.visual_dataloader(), **kwargs)
    def get_summary_imgs(self, pl_module, imgs, zoomfac=.5):
        all_images = []
        rows = len(imgs)
        for name, image_array in imgs:
            for img_name in image_array:
                img = image_array[img_name]
                all_images.append( torch.from_numpy(img) )
        # gather all images from all gpus
        # gathered = pl_module.all_gather(all_images)
        # if len(gathered[0].shape) == 4: # (world_size, H, W, C)
        #     world_size = gathered[0].shape[0]
        # else:                           # (H, W, C)
        #     world_size = 1
        #     gathered = [gimg[None,...] for gimg in gathered]
        # rows = world_size * rows
        # all_images = []
        # for i in range(world_size): # for each gpu
        #     for j in range(rows):
        #         all_images.append(gathered[j][i])
        # import pdb
        # pdb.set_trace()
        summary = visutil.imageGrid(all_images, shape=(rows, -1), zoomfac=zoomfac)
        return {self.classname: {"caption":self.classname, "image":summary}}
    def log_summary_images(self, trainer, pl_module, summary_imgs, x_axis="epoch"):
        # wandb logger
        import wandb
        for key in summary_imgs:
            t = summary_imgs[key]
            title   = key
            caption = t["caption"]
            image   = t["image"]
            #log_image(trainer, title, caption, image, trainer.global_step)
            #x_val = trainer.current_epoch if x_axis=="epoch" else trainer.global_step
            trainer.logger.experiment.log( \
                {title:[wandb.Image(image,caption=caption)], \
                    "epoch": trainer.current_epoch,
                    "global_step": trainer.global_step})


def null_logger(*args, **kwargs):
    return None
def get_debug_model(trainer, resume=False, device="cuda"):
    if resume == True:
        pl_model, train_dloader, val_dloader, test_dloader = trainer.test_mode(device=device)
    else:
        pl_model, train_dloader, val_dloader, test_dloader = trainer.test_mode(resume_from=None,device=device)
    train_dloader.num_workers = 0 # it will be very slow to invoke subprocesses (num_workers>0)
    val_dloader.num_workers = 0 # it will be very slow to invoke subprocesses (num_workers>0)
    test_dloader.num_workers  = 0
    return pl_model, train_dloader, val_dloader, test_dloader
def debug_model(trainer, resume=False, load_compute=False, load_visual=False, skip_batch_test=False):
    pl_model, train_dloader, val_dloader, test_dloader = get_debug_model(trainer, resume=resume)
    print("Test run train/val step")
    if skip_batch_test==False:
        test_batch(pl_model, train_dloader, val_dloader, test_dloader)

    visual_dloader = trainer.data_module.visual_dataloader()
    print(trainer.callbacks)
    for callback in trainer.callbacks:
        cb_name = callback.__class__.__name__
        if cb_name in ["ModelCheckpoint", "ProgressBar", "TQDMProgressBar", \
            "EarlyStopping", "LearningRateMonitor", "ModelSummary", "GradientAccumulationScheduler"]:
            continue
        # check if callback has process method
        if hasattr(callback, "process") == True:
            print("Start callback: ", cb_name)
            returns = callback.process(pl_model, visual_dloader, visual_summary=True, load_compute=load_compute, load_visual=load_visual, debug=True)
    print("Success")
    return pl_model, train_dloader, val_dloader, test_dloader
def test_batch(pl_model, train_dloader, val_dloader, test_dloader):
    exit_flag = False
    try:
        th_train_batch = ptutil.nps2ths(next(iter(train_dloader)), device="cuda")
        th_val_batch  = ptutil.nps2ths(next(iter(val_dloader)), device="cuda")
        origin_logger = pl_model.log
        pl_model.log = null_logger
        #optimizers, schedulers = pl_model.configure_optimizers()
        loss = pl_model.training_step(th_train_batch, batch_idx=0).detach().item()
        print(f"Batch {0} train loss:", loss)
        #for optimizer in optimizers:
        #    print("Testing optimization step of optimizer", optimizer )
        #    optimizer.step()
        loss = pl_model.validation_step(th_val_batch, batch_idx=0).detach().item()
        print(f"Batch {0} val loss:",   loss)
    except Exception as e:
        traceback.print_exc()
        print(e)
        print("Failed to run training/validation batch")
        
        exit_flag = True
    finally:
        pl_model.log = origin_logger
    if exit_flag == True:
        print("Exiting...")
        sys.exit(0)
