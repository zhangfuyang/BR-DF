import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from dataset import ReconDataset
from experiment import ReconVAEExperiment
from vae_model import VQVAE3D


def collate_fn(batch):
    solid_voxel = torch.stack([item[0] for item in batch])
    faces_voxel = torch.cat([item[1][:,None] for item in batch], dim=0)
    faces_bbox = torch.cat([item[2] for item in batch], dim=0)
    faces_num = [item[1].shape[0] for item in batch]
    filename = [item[3] for item in batch]
    return {'solid_voxel': solid_voxel, 
            'faces_voxel': faces_voxel, 
            'faces_bbox': faces_bbox,
            'filename': filename, 'faces_num': faces_num}

def load_model(model_class, model_config, pretrained_model_path=None, exclude_prefix='vae_model.'):
    model = model_class(**model_config)
    state_dict = torch.load(pretrained_model_path, map_location='cpu')['state_dict']
    # modify the key names
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(exclude_prefix):
            new_state_dict[k[len(exclude_prefix):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='vqvae/configs/recon.yaml')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

seed_everything(config['trainer_params']['manual_seed'], True)

solid_model = load_model(VQVAE3D, 
                        config['model_params'],
                        config['exp_params']['pretrained_solid_model_path'],
                        exclude_prefix='vae_model.')
face_model = load_model(VQVAE3D,
                        config['model_params'],
                        config['exp_params']['pretrained_face_model_path'],
                        exclude_prefix='vae_model.')
experiment = ReconVAEExperiment(config['exp_params'], solid_model, face_model)

val_dataset = ReconDataset(config['data_params']['train'])
print(len(val_dataset))
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=128,
    shuffle=False, num_workers=16,
    collate_fn=collate_fn)

trainer = pl.Trainer(
    accelerator=config['trainer_params']['accelerator'], max_epochs=config['trainer_params']['max_epochs'],
    precision=config['trainer_params']['precision'], num_nodes=config['trainer_params']['num_nodes'],
    devices=config['trainer_params']['devices'], strategy=config['trainer_params']['strategy'],
    log_every_n_steps=config['trainer_params']['log_every_n_steps'], 
    default_root_dir=config['trainer_params']['default_root_dir'])

# cp yaml file
if trainer.is_global_zero:
    os.makedirs(trainer.log_dir, exist_ok=True)
    os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

trainer.validate(experiment, val_dataloader)


