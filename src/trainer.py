import os
import sys
import glob
import shutil
import IPython
import argparse
import torch
from torch import cuda as torch_cuda
import torch.multiprocessing as mp

from omegaconf import OmegaConf

from lightning.pytorch import Trainer as plTrainer, loggers
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar

from xgutils import sysutil, optutil, datautil
from xgutils import qdaq
# set the default root to this trainer.py file's folder
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(FILE_DIR, os.path.pardir)

class Trainer():
    def default_opt(self):
        return dict(
            fast_dev_run=False,
            gpus=[0], # int for number, -1/'auto' for all, list for specified ones
            max_epochs = 100, min_epochs = 1, max_time=None, 
            copy_ckpt_from='',
            resume_from='',
            check_val_every_n_epoch=3,
            checkpoint_monitor='val/loss', # if =None, always save latest
            early_stop_patience=5,
            disable_auto_lr_scale=False,
            logger='wandb',  # ["wandb", "tensorboard"]
            logger_kwargs={},
            wandb_silent=False,
            auto_lr_find=False,
            gradient_clip_val=None,
            accelerator="auto",  # gpu
            strategy="ddp_find_unused_parameters_true", #  "ddp" or "dp"
            precision='32-true',
            use_tensorcores=True,
            seed=314,
            strict_loading=True,
            extra_kwargs=dict(),
        )

    def __init__(self, opt, root_dir=DEFAULT_ROOT, mode='test', gpus=[0]):
        if type(opt) is str:
            self.opt_string = opt
            opt = optutil.get_opt(opt, root_dir=root_dir,
                                  src_name='src')
        #self.opt = opt = argparse.Namespace(**opt)
        
        self.opt = opt = OmegaConf.create(opt)
        self.project_name = self.opt.project_name if hasattr(
            self.opt, "project_name") else "project"

        trainer_opt = self.trainer_opt = self.default_opt()
        sysutil.dictUpdate(self.trainer_opt, self.opt.pltrainer_opt)
        self.opt.pltrainer_opt = self.trainer_opt
        if gpus is not None and len(gpus) > 0:
            trainer_opt["gpus"] = gpus
        self.num_gpus = len(trainer_opt["gpus"])
        if len(trainer_opt["gpus"]) > 1:
            # use ddp_find_unused_parameters_true to avoid the error: RuntimeError: Expected to mark a variable ready only once.
            #trainer_opt["strategy"] = "ddp_find_unused_parameters_true"
            # The effective learning rate of multi gpu with ddp = lr / `num_gpus` (we need to tune lr according to effective batch_size)
            # So in order to keep the default lr, we need to multiply `num_gpus` see more details at https://github.com/Lightning-AI/lightning/discussions/3706#discussioncomment-238302
            # if self.trainer_opt["disable_auto_lr_scale"] == False and "optim_opt" in self.opt.pl_model_opt["kwargs"]:
            #     lr_name = "lr" if "lr" in self.opt.pl_model_opt["kwargs"]["optim_opt"] else "learning_rate"
            #     self.opt.pl_model_opt["kwargs"]["optim_opt"][lr_name] *= len(
            #         trainer_opt["gpus"])
            pass
        if IPython.get_ipython() is not None:
            trainer_opt["strategy"] = "auto"
        self.minfo = self.opt.meta_info

        # seed_everything(trainer_opt["seed"])
        if trainer_opt["use_tensorcores"]: # check more details here: https://github.com/Lightning-AI/lightning/issues/12997
            torch.set_float32_matmul_precision("medium") # enable tensorcores by not choosing "highest" precision

        self.load_model()
        if trainer_opt["strict_loading"] == False:
            self.model.strict_loading = False
        self.load_callbacks()

        print("???", trainer_opt['resume_from'], mode)
        if mode == 'train':
            if trainer_opt["copy_ckpt_from"] != '':
                self.copy_ckpt(trainer_opt["copy_ckpt_from"])
            self.resume_from_checkpoint = self.parse_resume(
                trainer_opt['resume_from'])
            logger = self.get_logger()
            optutil.expr_mkdirs(self.opt)
        else:
            resume = trainer_opt['resume_from']
            if resume == "restart":
                resume = "latest"
            self.resume_from_checkpoint = self.parse_resume(resume)
            logger = False
        self.mode = mode

        self.MainTrainerOptions = dict(
            fast_dev_run=trainer_opt['fast_dev_run'],
            max_epochs=trainer_opt['max_epochs'],
            devices=trainer_opt['gpus'],
            strategy=trainer_opt['strategy'],
            check_val_every_n_epoch=self.trainer_opt['check_val_every_n_epoch'],
            callbacks=self.callbacks,
            # gradient_clip_val=0.5, gradient_clip_algorithm="value",
        )
        self.OtherTrainerOptions = dict(
            logger=logger,
            #profiler="simple",
            accelerator=trainer_opt['accelerator'],
        )
        print("====== Max epochs =====",
              self.MainTrainerOptions["max_epochs"])
        self.trainer = \
            plTrainer(**self.MainTrainerOptions,
                      **self.OtherTrainerOptions,
                      **trainer_opt["extra_kwargs"]
                      )

    def load_model(self):
        self.opt.pl_model_opt["full_opt"] = self.opt
        self.opt.pl_model_opt["full_cfg"] = self.opt
        self.model = sysutil.instantiate_from_opt(self.opt.pl_model_opt)
        self.model_class = self.model.__class__
        self.data_module = sysutil.instantiate_from_opt(
            self.opt.datamodule_opt)

    def load_callbacks(self):
        self.callbacks, trainer_opt = [], self.trainer_opt
        if hasattr(self.opt, "callbacks"):
            for cb_name in self.opt.callbacks:
                cb_opt = self.opt.callbacks[cb_name]
                if "num_gpus" not in cb_opt:
                    cb_opt["num_gpus"] = self.num_gpus
                cb_class = sysutil.load_object(cb_opt["_target_"])
                cb_class_name = cb_class.__name__
                if "output_name" in cb_opt:
                    output_name = cb_opt["output_name"]
                else:
                    output_name = cb_class_name
                cb_kwargs = {"data_dir": os.path.join(
                    self.minfo["results_dir"], output_name)}
                cb_kwargs.update(cb_opt)
                print(cb_kwargs)
                callback = cb_class(**cb_kwargs)
                self.callbacks.append(callback)
                print("Callbacks", self.callbacks)
        # adding default callbacks
        self.checkpoint_callback = ModelCheckpoint(
            monitor=trainer_opt["checkpoint_monitor"],
            save_last=True,
            mode='min',
            filename="epoch{epoch:03d}-val_loss{val/loss:.4f}",
            auto_insert_metric_name=False,
            dirpath=self.minfo["checkpoints_dir"],
            verbose=True,
        )
        early_stop_callback = EarlyStopping(
            monitor='val/loss',
            min_delta=1e-6,
            patience= trainer_opt["early_stop_patience"],
            verbose=True,
            mode='min',
            strict=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        self.callbacks.append(self.checkpoint_callback)
        self.callbacks.append(early_stop_callback)
        self.callbacks.append(lr_monitor)
        self.callbacks.append(RichProgressBar())

    def get_logger(self):
        trainer_opt = self.trainer_opt
        if trainer_opt['logger'] == 'tensorboard':
            logger = loggers.TensorBoardLogger(self.minfo['logs_dir'])
            version = logger.experiment.log_dir.split('_')[-1]
        elif trainer_opt['logger'] == 'wandb':
            version = 0
            if "mode" in trainer_opt['logger_kwargs'] and trainer_opt['logger_kwargs']["mode"] == "disabled":
                print("Notice that the WANDB logger is disabled.")
            if "wandb_silent" in trainer_opt and trainer_opt['wandb_silent']:
                print("Notice that the WANDB logger is silent.")
                os.environ["WANDB_SILENT"] = "true"
            logger = loggers.WandbLogger(name=self.opt.expr_name, project=self.project_name, save_dir=self.minfo['logs_dir'], dir=self.minfo['logs_dir'], version=version,
                                         **trainer_opt['logger_kwargs'])
            #logger.save_dir = self.minfo['logs_dir']
            # print("!!!!!!", logger.experiment.dir)
            # logger.experiment.dir = self.minfo['logs_dir']
        else:
            logger = False
        return logger

    def parse_resume(self, ckpt):
        minfo = self.minfo
        if ckpt == '' or ckpt == 'restart':
            # if 'src/experiments/' not in minfo['expr_dir'] or minfo['expr_dir'].endswith('experiments') or minfo['expr_dir'].endswith('experiments/'):
            #     # prevent accidentally remove irrelevant files
            #     raise ValueError(f"Invalid resume_from: {minfo['expr_dir']}")
            # else:
            #     # if os.path.exists(minfo['expr_dir']):
            #     # print('==========================================')
            #     # print('Deleting old expr_dir and making a new one')
            #     # print('==========================================')
            #     # shutil.rmtree(minfo['expr_dir'])
            print("Starting training from scratch")
            return None
        if ckpt in ['latest', 'last', 'resume']:
            ckpts = glob.glob(os.path.join(minfo['checkpoints_dir'], '*'))
            print(minfo, 'ckpts')
            if len(ckpts) == 0:
                print(
                    f"Directory {minfo['checkpoints_dir']} has no checkpoints")
                return None
            latest_ckpt = max(ckpts, key=os.path.getctime)
            ckpt_path = latest_ckpt
        else:
            if ckpt[0] != '/':  # if it is relative path
                ckpt_path = os.path.join(minfo['checkpoints_dir'], ckpt)
            else:
                ckpt_path = ckpt
        print('Loading checkpoint: ', ckpt_path)
        return ckpt_path
    def train(self, resume_from=None):
        opt = self.opt
        minfo = opt.meta_info
        expr_dir = minfo['expr_dir']
        src_dir = minfo['src_dir']
        #optutil.dump(opt, os.path.join(expr_dir, 'config.yaml'))
        OmegaConf.save(opt, os.path.join(expr_dir, 'config.yaml'))
        src_backup = os.path.join(expr_dir, 'src')
        sysutil.makeArchive(minfo['src_dir'], src_backup)
        self.data_module.setup(stage=None) # get all dataloaders
        print("***********************", resume_from)
        if resume_from is None:
            resume_from = self.trainer_opt['resume_from']
            if resume_from == "restart":
                resume_from = None
            else:
                resume_from = self.parse_resume(resume_from)
        print("***********************+++++++++", resume_from)
        # if resume_from is not None:
        #     print("Train resume from", resume_from)
        #     self.model = self.model.__class__.load_from_checkpoint(resume_from)

        self.trainer.fit(self.model, datamodule=self.data_module, ckpt_path = resume_from)
        print("Model trained, best model path: ",
              self.checkpoint_callback.best_model_path)
        if self.checkpoint_callback.best_model_path is not None and self.checkpoint_callback.best_model_path != "":
            self.test(resume_from=self.checkpoint_callback.best_model_path)
        # backup after training
        if False:
            print('Finished training')
            save_path = os.path.join(
                minfo['experiments_dir'], minfo['session_name'])
            print('Save the experiment folder as %s' % (save_path))
            shutil.copytree(expr_dir, save_path)
            shutil.copytree(src_dir,  os.path.join(save_path, 'src'))
            print('Done.')

    def test(self, resume_from=None):
        self.data_module.prepare_data()
        self.data_module.setup(stage='test')
        if resume_from is None:
            resume_from = self.trainer_opt['resume_from']
            resume_from = self.parse_resume(resume_from)
        print("Test resume from", resume_from)
        if resume_from is not None and resume_from != "":
            self.model = self.model.__class__.load_from_checkpoint(resume_from, strict=False,
                map_location=torch.device('cpu'), )
        self.trainer.datamodule = self.data_module
        self.trainer.test(
            self.model, datamodule=self.data_module, ckpt_path=None)

    def test_mode(self, resume_from="last", device="cuda"):
        self.model = self.model.to(device)
        if resume_from is not None:
            resume_from = self.trainer_opt['resume_from']
            if resume_from == "restart":
                resume_from = "last"
            resume_from = self.parse_resume(resume_from)
            print("Test resume from", resume_from)
        if resume_from is not None:
            self.model = self.model.__class__.load_from_checkpoint(
                resume_from, strict=False, map_location=torch.device('cpu'), 
                **self.opt.pl_model_opt)

        self.data_module.prepare_data()
        self.data_module.setup(stage='test')
        train_dl, val_dl, test_dl = self.data_module.train_dataloader(), self.data_module.val_dataloader(), self.data_module.test_dataloader()

        # self.trainer.train_dataloaders = train_dl
        # self.trainer.val_dataloaders   = val_dl
        # self.trainer.test_dataloaders  = test_dl

        self.model = self.model.to(device)
        self.model.trainer = self.trainer
        self.model.eval()
        self.model.datamodule = self.data_module

        return self.model, train_dl, val_dl, test_dl 

    def run_callbacks(self):
        self.data_module.prepare_data()
        self.data_module.setup(stage=None)
        print("!!", self.resume_from_checkpoint)
        self.model = self.model_class.load_from_checkpoint(
            self.resume_from_checkpoint, strict=False)
        for callback in self.callbacks:
            if hasattr(callback, "post_training_process"):
                print("Run callback: ", callback)
                callback.post_training_process(
                    self.trainer, self.model, self.data_module)


class ExpJob(qdaq.Job):
    """ Assign a single GPU to each experiments
    """

    def __init__(self, opt):
        if type(opt) is str:
            opt = optutil.get_opt(
                opt, root_dir=DEFAULT_ROOT, src_name='shapeformer')
        self.opt = opt
        self.minfo = opt['meta_info']

    def run(self, cuda_device_id):
        # Get cuda device
        PID = os.getpid()
        print(
            f"Name: {self.opt['expr_name']} CUDA: {cuda_device_id}, PID: {PID}")
        if type(cuda_device_id) is int:
            cuda_device_id = [cuda_device_id]
        self.opt['pltrainer_opt']['gpus'] = cuda_device_id
        print("===========", cuda_device_id, "===========")
        trainer = Trainer(self.opt, mode='train', gpus=cuda_device_id)
        sys.stdout = open(os.path.join(
            self.minfo['logs_dir'], "stdout.out"), "w")
        trainer.train()
        return True


if __name__ == '__main__':
    # use spawn method to enable multi-processing for pytorch
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(add_help=False)
    #parser.add_argument('--option_path', required=True, type=str, help='path to project options')
    parser.add_argument('--opts', type=str, nargs='+',
                        help='path to project options')
    parser.add_argument('--gpus', type=int, nargs='*', help='gpus to use')
    parser.add_argument('--mode', type=str, default="train",
                        choices=["train", "test", "run"], help='train or run callbacks')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parsed = parser.parse_args()
    gpus = parsed.gpus
    if gpus is None or len(gpus) == 0:
        gpus = [0]  # , 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif gpus[0] == -1:
        gpus = list(range(torch_cuda.device_count()))
    #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(lambda x: str(x), gpus))
    print("**** INFO **** Choosing GPUS:", gpus)
    # gpus = list(range(len(gpus)))
    # print("**** INFO **** Resetting GPUS ids to:", gpus)

    assert (parsed.opts is not None) and len(parsed.opts) >= 1
    # If there are multiple option files specified, queue them according to gpus.
    if len(parsed.opts) == 1:
        

        print("**** INFO **** Running single opt file")
        with sysutil.debug(if_debug=parsed.debug==True):
            trainer = Trainer(parsed.opts[0], mode=parsed.mode, gpus=gpus)
            if parsed.mode == "train":
                func = trainer.train
            if parsed.mode == "test":
                func = trainer.test
            if parsed.mode == "run":
                func = trainer.run_callbacks
            func()
    else:
        print("**** INFO **** Multiple opt files: starting parallel jobs")
        exps = [ExpJob(opt) for opt in parsed.opts]
        qdaq.start(exps, gpus)
