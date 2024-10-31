import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from dataset import VoxelDataset
from experiment import VAEExperiment
from vae_model import VQVAE3D


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='vqvae/configs/vqvae_voxel_face_dist.yaml')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


model = VQVAE3D(**config['model_params'])
experiment = VAEExperiment(config['exp_params'], model)
# load pretrained model
if config['exp_params']['pretrained_model_path'] is not None:
    experiment.load_state_dict(
        torch.load(config['exp_params']['pretrained_model_path'], 
                   map_location='cpu')['state_dict'], strict=True)

train_dataset = VoxelDataset(config['data_params']['train'])
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config['data_params']['batch_size'], 
    shuffle=True, num_workers=config['data_params']['num_workers'])

val_dataset = VoxelDataset(config['data_params']['val'])
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=128,
    shuffle=True, num_workers=config['data_params']['num_workers'])

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=1, monitor='val_loss', mode='min', 
    save_last=True, filename='{epoch}-loss={train_loss:.4f}')

checkpoint_callback_last = pl.callbacks.ModelCheckpoint(
    filename='last-model-{epoch:02d}',
    save_last=True,
)

seed_everything(config['trainer_params']['manual_seed'], True)

trainer = pl.Trainer(
    accelerator=config['trainer_params']['accelerator'], max_epochs=config['trainer_params']['max_epochs'],
    precision=config['trainer_params']['precision'], num_nodes=config['trainer_params']['num_nodes'],
    devices=config['trainer_params']['devices'], strategy=config['trainer_params']['strategy'],
    log_every_n_steps=config['trainer_params']['log_every_n_steps'], 
    limit_val_batches=config['trainer_params']['limit_val_batches'],
    val_check_interval=config['trainer_params']['val_check_interval'],
    num_sanity_val_steps=config['trainer_params']['num_sanity_val_steps'],
    detect_anomaly=config['trainer_params']['detect_anomaly'],
    default_root_dir=config['trainer_params']['default_root_dir'],
    callbacks=[checkpoint_callback, checkpoint_callback_last])

# cp yaml file
if trainer.is_global_zero:
    os.makedirs(trainer.log_dir, exist_ok=True)
    os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

trainer.fit(experiment, train_dataloader, val_dataloader)

