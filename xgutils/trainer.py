import os
import shutil
import argparse

from pytorch_lightning import Trainer as plTrainer, loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import sys
sys.path.append("/studio/nnrecon/")
from nnrecon.models import find_model_using_name
#from nnrecon import options
from xgutils import sysutil, optutil
FILE_DIR            = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT       = os.path.join(FILE_DIR, os.path.pardir)

class Trainer():
    def default_opt(self):
        return dict(
            accelerator="ddp", #distributed_backend='dp',
            gpus=[2],
            resume_from='',
            check_val_every_n_epoch=3,
            disable_auto_lr_scale=True,
            logger='tensorboard',
            logger_kwargs={},
            auto_lr_find=False,
            gradient_clip_val=0,
            seed=314,
            save_top_k=1,
        )
    def __init__(self, opt, root_dir = DEFAULT_ROOT, mode='test', gpus=[0]):
        if type(opt) is str:
            opt = optutil.get_opt(opt, root_dir = root_dir, src_name='nnrecon')
        self.opt = opt = argparse.Namespace(**opt)
        self.project_name = self.opt.project_name if hasattr(self.opt, "project_name") else "nnrecon"

        trainer_opt = self.trainer_opt = self.default_opt()
        sysutil.dictUpdate(self.trainer_opt, self.opt.pltrainer_opt)
        if gpus is not None and len(gpus)>0:
            trainer_opt["gpus"] = gpus
        self.num_gpus = len(trainer_opt["gpus"])
        if len(trainer_opt["gpus"])==1:
            trainer_opt["accelerator"]="dp"
        else:
            # The effective learning rate of multi gpu with ddp = lr / `num_gpus` (we need to tune lr according to effective batch_size)
            # So in order to keep the default lr, we need to multiply `num_gpus`
            if self.trainer_opt["disable_auto_lr_scale"] == False:
                self.opt.pl_model_opt["kwargs"]["optim_opt"]["lr"] *= len(trainer_opt["gpus"])
        self.minfo = self.opt.meta_info

        seed_everything(trainer_opt["seed"])

        self.load_model()
        self.load_callbacks()


        if mode=='train':
            self.resume_from_checkpoint = self.parse_resume(trainer_opt['resume_from'])
            logger = self.get_logger()
            optutil.expr_mkdirs(self.opt.__dict__)
        else:
            resume = trainer_opt['resume_from']
            if resume == "restart":
                resume = "latest"
            self.resume_from_checkpoint = self.parse_resume(resume)
            logger = False
        self.mode = mode

        self.MainTrainerOptions=dict(
            max_epochs=trainer_opt['max_epochs'],
            gpus=trainer_opt['gpus'],
            auto_select_gpus=False, # very useful when gpu is in exclusive mode
            profiler="simple", 
            check_val_every_n_epoch=self.trainer_opt['check_val_every_n_epoch'],
            #auto_lr_find=self.trainer_opt['auto_lr_find'],
            terminate_on_nan=True,
            callbacks=self.callbacks,
            #progress_bar_refresh_rate=0,
            #train_percent_check=1.,
        )
        self.OtherTrainerOptions=dict(
            logger=logger,
            accelerator=trainer_opt['accelerator'],
            resume_from_checkpoint=self.resume_from_checkpoint,
        )
        print("!!!!!!!!!!!!!!! Max epochs", self.MainTrainerOptions["max_epochs"])
        self.trainer = \
            plTrainer(**self.MainTrainerOptions, 
                    **self.OtherTrainerOptions,
                )
                # auto_lr_find
        if mode=="train" and trainer_opt["accelerator"]=="dp" and \
            trainer_opt["auto_lr_find"]==True and trainer_opt["resume_from"]=="restart":
            lr_finder = trainer.tuner.lr_find(self.model, datamodule = self.datamodule)
            print(lr_finder.results)
            fig = lr_finder.plot(suggest=True)
            fig.show()
            new_lr = lr_finder.suggestion()
            print(new_lr)
    def find_lr(self):
        print("Finding optimal learning rate...")
        lr_finder = self.trainer.tuner.lr_find(self.model, datamodule = self.data_module, early_stop_threshold=10)
        #print(lr_finder.results)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        new_lr = lr_finder.suggestion()
        print("The suggested learning rate is:", new_lr)

    def load_model(self):
        self.model       = sysutil.instantiate_from_opt(self.opt.pl_model_opt)
        self.model_class = self.model.__class__
        self.data_module = sysutil.instantiate_from_opt(self.opt.datamodule_opt)
    def load_callbacks(self):
        self.callbacks, trainer_opt = [], self.trainer_opt
        if not hasattr(self.opt, "callbacks"):
            return
        for cb_name in self.opt.callbacks:
            cb_opt = self.opt.callbacks[cb_name]
            if "num_gpus" not in cb_opt["kwargs"]:
                cb_opt["kwargs"]["num_gpus"] = self.num_gpus
            cb_class = sysutil.load_object(cb_opt["class"])
            cb_class_name = cb_class.__name__
            if "output_name" in cb_opt["kwargs"]:
                output_name = cb_opt["kwargs"]["output_name"]
            else:
                output_name = cb_class_name
            cb_kwargs = {"data_dir": os.path.join(self.minfo["results_dir"], output_name)}
            cb_kwargs.update(cb_opt["kwargs"])
            print(cb_kwargs)
            callback = cb_class(**cb_kwargs)
            self.callbacks.append(callback)
        # adding default callbacks
        self.checkpoint_callback = ModelCheckpoint(
            #filepath=os.path.join(self.minfo['checkpoints_dir'],'{epoch:03d}'),
            monitor="val/loss",
            mode='min',
            filename="epoch{epoch:03d}-val_loss{val/loss:.4f}",
            auto_insert_metric_name=False,
            dirpath = self.minfo["checkpoints_dir"],
            save_top_k = trainer_opt["save_top_k"],
            save_last = True,
            verbose=True,
        )
        early_stop_callback = EarlyStopping(
            monitor='val/loss',
            min_delta=0.00001,
            patience=3,
            verbose=True,
            mode='min',
            strict=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        self.callbacks.append(self.checkpoint_callback)
        self.callbacks.append(early_stop_callback)
        self.callbacks.append(lr_monitor)
    def get_logger(self):
        trainer_opt = self.trainer_opt
        if trainer_opt['logger'] == 'tensorboard':
            logger = loggers.TensorBoardLogger(self.minfo['logs_dir'])
            version = logger.experiment.log_dir.split('_')[-1]
        elif trainer_opt['logger'] == 'wandb':
            logger = loggers.WandbLogger(name=self.opt.expr_name, project=self.project_name, 
                                            **trainer_opt['logger_kwargs'])
            version=0
        else:
            logger = False
        return logger
    def train(self):
        opt = self.opt
        minfo = opt.meta_info
        expr_dir = minfo['expr_dir']
        src_dir = minfo['src_dir']
        optutil.dump(opt.__dict__, os.path.join(expr_dir,'config.yaml') )
        src_backup = os.path.join(expr_dir,'src')
        sysutil.makeArchive(minfo['src_dir'], src_backup)
        self.data_module.setup()
        print("Test!!", len(self.data_module.train_set), len(self.data_module.val_set))
        self.trainer.fit(self.model, self.data_module)
        print("Model trained, best model path: ", self.checkpoint_callback.best_model_path)
        self.test(resume_from=self.checkpoint_callback.best_model_path) 
        # backup after training
        if False:
            print('Finished training')
            save_path = os.path.join(minfo['experiments_dir'], minfo['session_name'])
            print('Save the experiment folder as %s'%(save_path))
            shutil.copytree(expr_dir, save_path)
            shutil.copytree(src_dir,  os.path.join(save_path,'src'))
            print('Done.')
    def test(self, resume_from=None):
        self.data_module.prepare_data()
        self.data_module.setup()
        if resume_from is None:
            resume_from = self.trainer_opt['resume_from']
            if resume_from=="restart":
                resume_from = "last"
            resume_from = self.parse_resume(resume_from)
        print("Test resume from", resume_from)
        self.model = self.model.load_from_checkpoint(resume_from)
        self.trainer.datamodule = self.data_module
        self.trainer.test(self.model, self.data_module.test_dataloader(), ckpt_path=None)
    def test_mode(self, resume_from="last", device="cuda"):
        if resume_from is None:
            resume_from=None
        else:
            resume_from = self.trainer_opt['resume_from']
            if resume_from=="restart":
                resume_from = "last"
            resume_from = self.parse_resume(resume_from)
            print("Test resume from", resume_from)
            self.model = self.model.load_from_checkpoint(resume_from, **self.opt.pl_model_opt["kwargs"])
        self.model = self.model.to(device)
        self.model.eval()
        self.data_module.prepare_data()
        self.data_module.setup()
        return self.model, self.data_module.train_dataloader(), self.data_module.val_dataloader(), self.data_module.test_dataloader()
    def parse_resume(self, ckpt):
        minfo = self.minfo
        if ckpt == '' or ckpt == 'restart':
            if 'nnrecon/experiments/' not in minfo['expr_dir'] or minfo['expr_dir'].endswith('experiments') or minfo['expr_dir'].endswith('experiments/'): 
                raise ValueError(f"Invalid resume_from: {minfo['expr_dir']}") # prevent accidentally remove irrelevant files
            else:
                #if os.path.exists(minfo['expr_dir']):
                    # print('==========================================')
                    # print('Deleting old expr_dir and making a new one')
                    # print('==========================================')
                    # shutil.rmtree(minfo['expr_dir'])
                pass
            return None
        if ckpt in ['latest', 'last']:
            ckpts = glob.glob( os.path.join(minfo['checkpoints_dir'], '*') )
            print(minfo,'ckpts')
            if len(ckpts)==0:
                print(f"Directory {minfo['checkpoints_dir']} has no checkpoints")
                return None
            latest_ckpt = max(ckpts, key=os.path.getctime)
            ckpt_path = latest_ckpt
        else:
            if ckpt[0]!='/': # if it is relative path
                ckpt_path = os.path.join(minfo['checkpoints_dir'], ckpt)
            else:
                ckpt_path = ckpt
        print('Loading checkpoint: ', ckpt_path)
        return ckpt_path
    def run_callbacks(self):
        self.model = self.model_class.load_from_checkpoint(self.resume_from_checkpoint)
        for callback in self.callbacks:
            if hasattr(callback, "post_training_process"):
                print("Run callback: ", callback)
                callback.post_training_process(self.trainer, self.model, self.data_module)

from xgutils import qdaq
import sys
class ExpJob(qdaq.Job):
    def __init__(self, opt):
        if type(opt) is str:
            opt = optutil.get_opt(opt, root_dir = DEFAULT_ROOT, src_name='nnrecon')
        self.opt = opt
        self.minfo = opt['meta_info']
    def run(self, cuda_device_id):
        # Get cuda device
        PID = os.getpid()
        print(f"Name: {self.opt['expr_name']} CUDA: {cuda_device_id}, PID: {PID}")
        if type(cuda_device_id) is int:
            cuda_device_id = [cuda_device_id]
        self.opt['pltrainer_opt']['gpus'] = cuda_device_id
        print("===========", cuda_device_id, "===========")
        #print(self.opt)
        #return True
        trainer = Trainer(self.opt, mode='train', gpus = cuda_device_id)
        sys.stdout = open(os.path.join(self.minfo['logs_dir'], "stdout.out"), "w")
        trainer.train()
        return True
import glob
import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(add_help=False)
    #parser.add_argument('--option_path', required=True, type=str, help='path to project options')
    parser.add_argument('--opts', type=str, nargs='+', help='path to project options')
    parser.add_argument('--gpus', type=int, nargs='*', help='gpus to use')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test", "run"], help='train or run callbacks')
    parsed=parser.parse_args()
    gpus = parsed.gpus
    if gpus is None or len(gpus)==0:
        gpus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join( map(lambda x:str(x), gpus))
    gpus = list(range(len(gpus)))

    if len(parsed.opts)==1:
        trainer = Trainer(parsed.opts[0], mode='train', gpus=gpus)
        if parsed.mode == "train":
            trainer.train()
        if parsed.mode == "test":
            trainer.test()
        if parsed.mode == "run":
            trainer.run_callbacks()
    else:
        exps = [ExpJob(opt) for opt in parsed.opts]
        qdaq.start(exps, gpus)
