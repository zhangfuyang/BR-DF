import os
import torch
import yaml
import argparse
from diffusers import DDPMScheduler, PNDMScheduler, DDIMScheduler
import matplotlib.pyplot as plt
import numpy as np
import mcubes
import trimesh
from diffusion_model import Solid3DModel
from tqdm import tqdm
import pickle
import sys
from utils import draw_cuboid, base_color
from optimize import brep_process

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vqvae.vae_model import VQVAE3D
from bbox.network import BBoxNet


parser = argparse.ArgumentParser()
parser.add_argument('--bbox_config', type=str, default='checkpoints/deepcad/bbox_config.yaml')
parser.add_argument('--bbox_checkpoint', type=str, default='checkpoints/deepcad/bbox_last.ckpt')
parser.add_argument('--face_solid_config', type=str, default='checkpoints/deepcad/bbox_sdf_diffusion_config.yaml')
parser.add_argument('--face_solid_checkpoint', type=str, default='checkpoints/deepcad/bbox_sdf_diffusion_last.ckpt')
parser.add_argument('--output_dir', type=str, default='bbox_sdf_diffusion/output_deepcad')
parser.add_argument('--fast_sample', action='store_true', default=False)
args = parser.parse_args()

batch_size = 8
num_bbox = 50
bbox_threshold = 0.04

### 1. load bbox model ###
with open(args.bbox_config, 'r') as f:
    config = yaml.safe_load(f)
bbox_net = BBoxNet(config['network_params'])
# load checkpoint
state_dict = torch.load(args.bbox_checkpoint, map_location='cpu')['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('network.'):
        new_state_dict[k[8:]] = v
    else:
        new_state_dict[k] = v
bbox_net.load_state_dict(new_state_dict, strict=True)
bbox_net = bbox_net.to('cuda').eval()

### 2. load face_solid model ###
with open(args.face_solid_config, 'r') as f:
    config = yaml.safe_load(f)
face_solid_net = Solid3DModel(config['model_params'])
# load checkpoint
state_dict = torch.load(args.face_solid_checkpoint, map_location='cpu')['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('face_model') or k.startswith('sdf_model'):
        continue
    if k.startswith('diffusion_model.'):
        new_state_dict[k[16:]] = v
    else:
        new_state_dict[k] = v
face_solid_net.load_state_dict(new_state_dict, strict=True)
face_solid_net = face_solid_net.to('cuda').eval()

### 3. load face & solid VAE model ###
# load face model
face_yaml = config['exp_params']['face_model']['config']
with open(face_yaml, 'r') as f:
    face_config = yaml.safe_load(f)['model_params']
face_model = VQVAE3D(**face_config)
state_dict = torch.load(
    config['exp_params']['face_model']['pretrained_path'], map_location='cpu')['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('vae_model.'):
        new_state_dict[k[10:]] = v
    else:
        new_state_dict[k] = v
face_model.load_state_dict(new_state_dict, strict=True)
face_model = face_model.to('cuda').eval()

# load solid model
solid_yaml = config['exp_params']['sdf_model']['config']
with open(solid_yaml, 'r') as f:
    solid_config = yaml.safe_load(f)['model_params']
solid_model = VQVAE3D(**solid_config)
state_dict = torch.load(
    config['exp_params']['sdf_model']['pretrained_path'], map_location='cpu')['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('vae_model.'):
        new_state_dict[k[10:]] = v
    else:
        new_state_dict[k] = v
solid_model.load_state_dict(new_state_dict, strict=True)
solid_model = solid_model.to('cuda').eval()

with open(config['exp_params']['latent_std_mean_path'], 'rb') as f:
    mean_std = pickle.load(f)
solid_mean, solid_std = mean_std['solid_mean'], mean_std['solid_std']
face_mean, face_std = mean_std['face_mean'], mean_std['face_std']

########## sampling ##########

pndm_scheduler = PNDMScheduler(
    num_train_timesteps=1000,
    beta_schedule='linear',
    prediction_type='epsilon',
    beta_start=0.0001,
    beta_end=0.02,)

ddpm_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule='linear',
    prediction_type='epsilon',
    beta_start=0.0001,
    beta_end=0.02,
    clip_sample=True,
    clip_sample_range=1,)

### 1. sample bbox ###
with torch.no_grad():
    bbox = torch.randn(batch_size, num_bbox, 6).float().to('cuda')

    ddpm_scheduler.set_timesteps(1000)
    for t in tqdm(ddpm_scheduler.timesteps):
        timesteps = torch.full((batch_size,), t, device='cuda').long()
        pred = bbox_net(bbox, timesteps)
        bbox = ddpm_scheduler.step(pred, t, bbox).prev_sample

# remove duplicate bbox
bbox_raw = bbox.cpu().numpy() # (bs, num_bbox, 6) 
# (bs, num_bbox, 2, 3) 

bbox_deduplicate = []
for b_i in range(batch_size):
    x = np.round(bbox_raw[b_i], 4)
    non_repeat = x[:1]
    non_repeat_idx = [0]
    for i in range(1, len(x)):
        diff = np.max(np.abs(non_repeat - x[i]), -1)
        same = diff < bbox_threshold
        if same.sum()>=1:
            continue
        non_repeat = np.concatenate([non_repeat, x[i][None]], axis=0)
        non_repeat_idx.append(i)
    
    deduplicate = bbox_raw[b_i, non_repeat_idx]
    bbox_deduplicate.append(deduplicate)

bbox = bbox_deduplicate
face_num_list = [len(b) for b in bbox]
max_face_num = max(face_num_list)
bbox_merge = []
for i, bbox_i in enumerate(bbox):
    bbox_i = torch.cat([
        torch.from_numpy(bbox_i).float(), 
        torch.zeros((max_face_num-face_num_list[i], 6))], 
    dim=0)
    bbox_merge.append(bbox_i)
bbox_merge = torch.stack(bbox_merge, dim=0) # (b,m,6)

# mask
mask = torch.zeros((batch_size, max_face_num))
for i, face_num in enumerate(face_num_list):
    mask[i, :face_num] = 1

### 2. sample face_solid ###
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.02,
    beta_schedule='scaled_linear',
    clip_sample=False,
    set_alpha_to_one=False,
    prediction_type=config['exp_params']['prediction_type'],
    )
num_bbox = bbox_merge.shape[1]
with torch.no_grad():
    bbox = bbox_merge.to('cuda') # (bs, num_bbox, 6)
    mask = mask.to('cuda')

    z_solid = torch.randn(batch_size, 1, 4,4,4,4).to('cuda')
    z_faces = torch.randn(batch_size, num_bbox,4,4,4,4).to('cuda')

    ddim_scheduler.set_timesteps(200)
    timesteps = ddim_scheduler.timesteps
    for i, t in tqdm(enumerate(timesteps)):
        timestep = torch.cat([t.unsqueeze(0)]*batch_size, dim=0).to('cuda')
        model_output_solid, model_output_faces = face_solid_net(
            z_solid, z_faces, bbox, mask, timestep)
        z_solid = ddim_scheduler.step(model_output_solid, t, z_solid).prev_sample
        z_faces = ddim_scheduler.step(model_output_faces, t, z_faces).prev_sample

# render

for i in range(batch_size):
    print(f'sample {i}')
    save_root = os.path.join(args.output_dir, f'{i:03d}')
    os.makedirs(save_root, exist_ok=True)
    
    solid_latent = z_solid[i] # (1,4,4,4,4)
    face_latent = z_faces[i] # (num_bbox,4,4,4,4)
    
    solid_latent = solid_latent * solid_std + solid_mean
    face_latent = face_latent * face_std + face_mean

    with torch.no_grad():
        solid_voxel = solid_model.quantize_decode(solid_latent)[0,0] # (n,n,n)
        face_voxel = face_model.quantize_decode(face_latent)[mask[i]==1,0] # (m,n,n,n)

    bbox_i = bbox[i, mask[i]==1].cpu().numpy()
    solid_voxel_i = solid_voxel.cpu().numpy()
    face_voxel_i = face_voxel.cpu().numpy()
    solid_voxel_i = solid_voxel_i.transpose(1,0,2)
    face_voxel_i = face_voxel_i.transpose(0,2,1,3)

    # draw bbox
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for b_idx, b in enumerate(bbox_i):
        size = b[3:] - b[:3]
        size[size < 0.1] = 0.1
        ori = b[:3]
        draw_cuboid(ax, ori, size, base_color[b_idx % len(base_color)] / 255)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    plt.savefig(os.path.join(save_root, 'bbox.png'))
    plt.close()

    vertices, triangle = mcubes.marching_cubes(solid_voxel_i, 0)
    vertices = vertices / solid_voxel_i.shape[0] * 2 - 1 # [-1,1] (k,3)
    solid_mesh = trimesh.Trimesh(vertices, triangle)
    solid_mesh.export(os.path.join(save_root, 'solid.obj'))

    # save each face
    all_pc = None
    for face_i in range(face_voxel_i.shape[0]):
        face = face_voxel_i[face_i]
        points = np.where(face < 0.2)
        points = np.array(points).T
        pc = trimesh.points.PointCloud(points)
        if pc.vertices.shape[0] == 0:
            continue
        pc.visual.vertex_colors = base_color[face_i % len(base_color)]
        #pc.export(os.path.join(save_root, f'face_{face_i}_only.obj'))

        all_pc = pc if all_pc is None else all_pc + pc

    all_pc.export(os.path.join(save_root, 'all_face.obj'))


    # save solid_voxel_i and face_voxel_i
    data = {
        'v_sdf': solid_voxel_i,
        'f_udf': face_voxel_i.transpose(1,2,3,0)
    }
    np.save(os.path.join(save_root, 'data.npy'), data)
    ##### process #####
    save_root = os.path.join(args.output_dir, f'{i:03d}', 'process')
    os.makedirs(save_root, exist_ok=True)
    brep_process(solid_voxel_i, face_voxel_i.transpose(1,2,3,0), save_root, save_rot_images=True)


    
