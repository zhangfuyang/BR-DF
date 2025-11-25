# B-Rep Distance Functions (BR-DF) <br> How to Represent a B-Rep Model by Volumetric Distance Functions?

# The code has not been fully cleaned, and the current README is not yet readable. The final version will be released by the end of this year.

[![arXiv](https://img.shields.io/badge/📃-arXiv%20-red.svg)](https://arxiv.org)
[![webpage](https://img.shields.io/badge/🌐-Website%20-blue.svg)](https://zhangfuyang.github.io/brdf/) 

![alt brdf](https://zhangfuyang.github.io/brdf/assets/images/teaser_v3.jpg)

> We present a novel geometric representation for CAD Boundary Representation (B-Rep) based on volumetric distance functions, dubbed B-Rep Distance Functions (BR-DF). BR-DF encodes the surface mesh geometry of a CAD model as signed distance function (SDF). B-Rep vertices, edges, faces and their topology information are encoded as per-face unsigned distance functions (UDFs). An extension of the classical Marching Cubes algorithm converts BR-DF directly into watertight CAD B-Rep model.

## Requirements
- Linux
- Python 3.9
- CUDA 11.8 
- PyTorch 2.2 
- Diffusers 0.27

## Data
Download [ABC](https://archive.nyu.edu/handle/2451/43778) STEP files (100 folders). 

Extract SDF voxel and UDF voxels from STEP files.

    bash data_process_script.sh

## Training
1. Download checkpoints "abc folder" from the above link.
2. Download "pkl.tar" from https://1sfu-my.sharepoint.com/:f:/g/personal/fuyangz_sfu_ca/EoBgkMc1LZZLkCFsQKFV2B0Bjnr5QLuop76jYwTpK3NyjA?e=cFBB6n. This is the pre computed latent of all abc data.
3. unzip pkl.tar and put at the location: brep_proj/data/latent_cache/pkl
4. run: python bbox_sdf_diffusion/run.py --config bbox_sdf_diffusion/train_large.yaml

Inside yaml, trainer_params->devices is the number of GPUs, trainer_params->num_nodes is the number of cluster nodes.


## Testing

    python bbox_sdf_diffusion/sample.py

## Pretrained Checkpoint

download checkpoint folder from https://1sfu-my.sharepoint.com/:f:/g/personal/fuyangz_sfu_ca/EjjVLHgS1UVElW46mgsfFj8BAhRcTz_2wuxowGjhuBbR-w?e=FfF5WN

Then run,
```
python bbox_sdf_diffusion/sample.py
```

