# An Object is Worth 64x64 Pixels: Generating 3D Object via Image Diffusion

**This repository is the official repository of the paper, *An Object is Worth 64x64 Pixels: Generating 3D Object via Image Diffusion*.**

[Xinggaung Yan](http://yanxg.art)<sup>1</sup>,
[Han-Hung Lee](https://hanhung.github.io/)<sup>1</sup>,
[Ziyu Wan](http://raywzy.com/)<sup>2</sup>,
[Angel X. Chang](https://angelxuanchang.github.io/)<sup>1,3</sup>

<sup>1</sup>Simon Fraser University, <sup>2</sup>City University of Hong Kong, <sup>3</sup>Canada-CIFAR AI Chair, Amii

### [Project Page](https://omages.github.io/) | [Paper (ArXiv)](https://arxiv.org/abs/) 
<!-- | [Twitter thread](https://twitter.com/yan_xg/status/1539108339422212096) -->
<!-- | [Pre-trained Models](https://www.dropbox.com/s/we886b1fqf2qyrs/ckpts_ICT.zip?dl=0) :fire: |  -->

<img src='assets/fig_teaser.png'/>

<!-- https://user-images.githubusercontent.com/5100481/150949433-40d84ed1-0a8d-4ae4-bd53-8662ebd669fe.mp4 -->

https://github.com/user-attachments/assets/e77098d1-7128-4510-913f-5af302544850


## :hourglass_flowing_sand: UPDATES
- [ ] Source code and data, coming soon!

<!-- ## Installation
The code is tested in docker enviroment [pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel](https://hub.docker.com/layers/pytorch/pytorch/pytorch/1.6.0-cuda10.1-cudnn7-devel/images/sha256-ccebb46f954b1d32a4700aaeae0e24bd68653f92c6f276a608bf592b660b63d7?context=explore).
The following are instructions for setting up the environment in a Linux system from scratch.
You can also directly pull our provided docker environment: `sudo docker pull qheldiv/shapeformer`
Or build the docker environment by yourself with the setup files in the `Docker` folder.

First, clone this repository with submodule xgutils. [xgutils](https://github.com/QhelDIV/xgutils.git) contains various useful system/numpy/pytorch/3D rendering related functions that will be used by ShapeFormer.

      git clone --recursive https://github.com/QhelDIV/ShapeFormer.git

Then, create a conda environment with the yaml file. (Sometimes the conda is very slow to solve the complex dependencies of this environment, so [mamba](https://mamba.readthedocs.io/en/latest/index.html) is highly recommended)

      conda env create -f environment.yaml
      conda activate shapeformer

Next, we need to install torch_scatter through this command

      pip install torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.7.0+cu101.html

## Demo

First, download the pretrained model from this google drive [URL](https://drive.google.com/file/d/1QmR27nHcLmzFfyvxs3NH7pzUmbVATt4f/view?usp=sharing) and extract the content to experiments/

Then run the following command to test VQDIF. The results are in `experiments/demo_vqdif/results`

      python -m shapeformer.trainer --opts configs/demo/demo_vqdif.yaml --gpu 0 --mode "run"

Run the following command to test ShapeFormer for shape completion. The results are in `experiments/demo_shapeformer/results`

      python -m shapeformer.trainer --opts configs/demo/demo_shapeformer.yaml --gpu 0 --mode "run"

## Dataset

We use the dataset from [IMNet](https://github.com/czq142857/IM-NET#datasets-and-pre-trained-weights), which is obtained from [HSP](https://github.com/chaene/hsp).

The dataset we adopted is a downsampled version (64^3) from these dataset (which is 256 resolution).
Please download our processed dataset from this google drive [URL](https://drive.google.com/file/d/1HUbI45KmXCDJv-YVYxRj-oSPCp0D0xLh/view?usp=sharing).
And then extract the data to `datasets/IMNet2_64/`.

To use the full resolution dataset, please first download the original IMNet and HSP datasets, and run the `make_imnet_dataset` function in `shapeformer/data/imnet_datasets/imnet_datasets.py`

### D-FAUST Human Dataset
We also provide the scripts for process the D-FAUST human shapes. 
First, download the official D-FAUST dataset from this [link](https://dfaust.is.tuebingen.mpg.de/download.php) and extract to `datasets/DFAUST`
Then, execute the following lines to generate obj files and generate sdf samples for the human meshes.

      cd shapeformer/data/dfaust_datasets/datagen
      python generate_dfaust_obj_runfile.py
      bash generate_dfaust_obj_all.sh
      python generate_dfaust_sdf_samples.py

## Usage


First, train VQDIF-16 with 

      python -m shapeformer.trainer --opts configs/vqdif/shapenet_res16.yaml --gpu 0

After VQDIF is trained, train ShapeFormer with

      python -m shapeformer.trainer --opts configs/shapeformer/shapenet_scale.yaml --gpu 0

For testing, you just need to append `--mode test` to the above commands.
And if you only want to run callbacks (such as visualization/generation), set the mode to `run`

There is a visualization callback for both VQDIF and ShapeFormer, who will call the model to obtain 3D meshes and render them to images. The results will be save in `experiments/$exp_name$/results/$callback_name$/`
The callbacks will be automatically called during training and testing, so to get the generation results you just need to test the model.

ALso notice that in the configuration files batch sizes are set to very small so that the model can run on a 12GB memory GPU. You can tune it up if your GPU has a larger memory.

### Multi-GPU
Notice that to use multiple GPUs, just specify the GPU ids. For example `--gpu 0 1 2 4` is to use the 0th, 1st, 2nd, 4th GPU for training. Inside the program their indices will be mapped to 0 1 2 3 for simplicity.

## Frequently Asked Questions

*What is the meaning of the variables Xbd, Xtg, Ytg... ?*

Here is a brief description of the variable names:

> `tg` stands for `target`, which is the samples (probes) of the target occupancy fields.
> `bd`, or `boundary` stands for the points sampled from the shape surface.
> `ct` stands for `context`, which is the partial point cloud that we want to complete.
> `X` stands for point coordinate.
> `Y` stands for the occupancy value of the point coordinate.

The `target` and `context` names come from the field of meta-learning.

Notice that the `Ytg` in the hdf5 file stands for the occupancy value of the probes `Xtg`.
In the case of `IMNET2_64`, `Xtg` is the collection of the 64-grid coordinates, which has the shape of `(64**3, 3)` and `Ytg` is the corresponding occupancy value.
It is easy to visualize the shape with marching cubes if `Xtg` is points of a grid. But you can use arbitrarily sampled points as `Xtg` and `Ytg` for training.

*How can I evaluate the ShapeFormer?*

[Here](https://drive.google.com/file/d/1KjbFUuxTWrZ97Cz8ZlFoOB3gDGCyuwt-/view?usp=share_link) is an incomplete collection of evaluation code of ShapeFormer.  -->

## :notebook_with_decorative_cover: Citation

If you find our work useful for your research, please consider citing the following papers :)

```bibtex
@misc{
}
```
<!-- 
## 📢: Shout-outs
The architecture of our method is inspired by [ConvONet](https://github.com/autonomousvision/convolutional_occupancy_networks), [Taming-transformers](https://github.com/CompVis/taming-transformers) and [DCTransformer](https://github.com/benjs/DCTransformer-PyTorch).
Thanks to the authors.

Also, make sure to check this amazing transformer-based image completion project([ICT](https://github.com/raywzy/ICT))! -->

## :email: Contact

This repo is currently maintained by Xingguang ([@qheldiv](https://github.com/qheldiv)) and is for academic research use only. Discussions and questions are welcome via qheldiv@gmail.com. 
