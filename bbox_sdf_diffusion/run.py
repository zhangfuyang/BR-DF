import os
import torch
torch.set_float32_matmul_precision('medium')
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from dataset import LatentDataset
from experiment import DiffusionExperiment
from diffusion_model import Solid3DModel
from utils import load_model
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vqvae')))
from vae_model import VQVAE3D


def collate_fn(batch):
    bs = len(batch)
    # solid_voxel: bs, 1, c, n,n,n
    solid_voxel = torch.stack([item['solid_voxel'] for item in batch])

    # faces_voxel: bs, M, c, n,n,n
    face_num_list = [item['face_voxel'].shape[0] for item in batch]
    max_face_num = max(face_num_list)
    faces_merge = []
    for i in range(bs):
        faces_voxel_i = batch[i]['face_voxel'] # m, c, n,n,n
        faces_pad = torch.cat([
            faces_voxel_i,
            torch.zeros((max_face_num - faces_voxel_i.shape[0], *faces_voxel_i.shape[1:]))
        ], dim=0) # M, c, n,n,n
        faces_merge.append(faces_pad)
    faces_voxel = torch.stack(faces_merge, dim=0) # bs, M, c, n,n,n

    # faces_bbox: bs, M, 6
    bbox_merge = []
    for i in range(bs):
        faces_bbox_i = batch[i]['face_bboxes'] # m, 6
        faces_pad = torch.cat([
            faces_bbox_i,
            torch.zeros((max_face_num - faces_bbox_i.shape[0], *faces_bbox_i.shape[1:]))
        ], dim=0) # M, 6
        bbox_merge.append(faces_pad)
    faces_bbox = torch.stack(bbox_merge, dim=0) # bs, M, 6

    # mask: bs, M
    mask = torch.zeros((bs, max_face_num))
    for i, face_num in enumerate(face_num_list):
        mask[i, :face_num] = 1

    faces_num = [item['face_voxel'].shape[0] for item in batch]
    filename = [item['filename'] for item in batch]
    return {'solid_voxel': solid_voxel, 
            'faces_voxel': faces_voxel, 
            'faces_bbox': faces_bbox,
            'mask': mask,
            'filename': filename, 'faces_num': faces_num}


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='bbox_sdf_diffusion/train.yaml')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


face_model = load_model(VQVAE3D, 
                        config['exp_params']['face_model']['config'], 
                        config['exp_params']['face_model']['pretrained_path'],
                        exclude_prefix='vae_model.')
sdf_model = load_model(VQVAE3D, 
                        config['exp_params']['sdf_model']['config'], 
                        config['exp_params']['sdf_model']['pretrained_path'],
                        exclude_prefix='vae_model.')
if config['model_params']['class_type'] == 'Solid3DModel':
    ModelClass = Solid3DModel
model = ModelClass(config['model_params'])

experiment = DiffusionExperiment(config['exp_params'], model, face_model, sdf_model)
# load pretrained model
if config['exp_params']['pretrained_model_path'] is not None:
    state_dict = torch.load(config['exp_params']['pretrained_model_path'],
                            map_location='cpu')['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('face_model.') or k.startswith('sdf_model.'):
            continue
        new_state_dict[k] = v
    experiment.load_state_dict(new_state_dict, strict=False)

DataClass = LatentDataset
train_dataset = DataClass(config['data_params'])
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config['data_params']['train_batch_size'], 
    shuffle=True, num_workers=config['data_params']['num_workers'],
    collate_fn=collate_fn)

val_dataset = DataClass(config['data_params'])
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=8, 
    shuffle=True, num_workers=config['data_params']['num_workers'],
    collate_fn=collate_fn)

seed_everything(config['trainer_params']['manual_seed'], True)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=1, monitor='train_loss', mode='min', 
    save_last=True, filename='{epoch}-loss={train_loss:.4f}')

checkpoint_callback_last = pl.callbacks.ModelCheckpoint(
    filename='last-model-{epoch:02d}',
    save_last=True,
)

lr_monitor = pl.callbacks.LearningRateMonitor()

trainer = pl.Trainer(
    accelerator=config['trainer_params']['accelerator'], max_epochs=config['trainer_params']['max_epochs'],
    precision=config['trainer_params']['precision'], num_nodes=config['trainer_params']['num_nodes'],
    devices=config['trainer_params']['devices'], strategy=config['trainer_params']['strategy'],#'ddp_find_unused_parameters_true', #config['trainer_params']['strategy'],
    log_every_n_steps=config['trainer_params']['log_every_n_steps'], 
    limit_val_batches=config['trainer_params']['limit_val_batches'],
    val_check_interval=config['trainer_params']['val_check_interval'],
    num_sanity_val_steps=config['trainer_params']['num_sanity_val_steps'],
    detect_anomaly=config['trainer_params']['detect_anomaly'],
    default_root_dir=config['trainer_params']['default_root_dir'],
    callbacks=[checkpoint_callback, checkpoint_callback_last, lr_monitor],
    accumulate_grad_batches=config['trainer_params']['accumulate_grad_batches'],
    gradient_clip_val=config['trainer_params']['gradient_clip_val'],
    check_val_every_n_epoch=None
    )

# cp yaml file
if trainer.is_global_zero:
    os.makedirs(trainer.log_dir, exist_ok=True)
    os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

trainer.fit(experiment, train_dataloader, val_dataloader)


