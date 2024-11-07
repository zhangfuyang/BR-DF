import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from network import BBoxNet
from dataset import Dataset
from diffusers import DDPMScheduler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

class Trainer(pl.LightningModule):
    def __init__(self, config, network):
        super(Trainer, self).__init__()
        self.config = config
        self.network = network

        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,)
    
    def training_step(self, batch, batch_idx):
        device = batch.device
        timesteps = torch.randint(0, 1000, (batch.shape[0],), device=device).long()
        noise = torch.randn_like(batch)
        x_noisy = self.scheduler.add_noise(batch, noise, timesteps)

        pred_noise = self.network(x_noisy, timesteps)
        loss = torch.nn.functional.mse_loss(pred_noise, noise)

        self.log('loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        device = batch.device

        batch_size = 10
        bbox = torch.randn(batch_size, *batch.shape[1:]).float().to('cuda')
        self.scheduler.set_timesteps(1000)
        for t in tqdm(self.scheduler.timesteps):
            timesteps = torch.full((batch_size,), t, device='cuda').long()
            pred = self.network(bbox, timesteps)
            bbox = self.scheduler.step(pred, t, bbox).prev_sample
        
        bbox = bbox.detach().cpu().numpy()
        # render
        for batch_i in range(batch_size):
            b = bbox[batch_i]
            x = np.round(b, 4)

            non_repeat = x[:1]
            non_repeat_idx = [0]
            for i in range(1, len(x)):
                diff = np.max(np.abs(non_repeat - x[i]), -1)
                same = diff < 0.05
                if same.sum()>=1:
                    continue
                non_repeat = np.concatenate([non_repeat, x[i][None]], axis=0)
                non_repeat_idx.append(i)
            
            x = x[non_repeat_idx]
            if x.shape[0] > 15:
                continue
    
            base_color = np.array(
                [[255,   0,  0, 255],  # Red
                [  0, 255,   0, 255],  # Green
                [  0,   0, 255, 255],  # Blue
                [255, 255,   0, 255],  # Yellow
                [  0, 255, 255, 255],  # Cyan
                [255,   0, 255, 255],  # Magenta
                [255, 165,   0, 255],  # Orange
                [128,   0, 128, 255],  # Purple
                [255, 192, 203, 255],  # Pink
                [128, 128, 128, 255],  # Gray
                [210, 245, 60, 255], # Lime
                [170, 110, 40, 255], # Brown
                [128, 0, 0, 255], # Maroon
                [0, 128, 128, 255], # Teal
                [0, 0, 128, 255], # Navy
                ],
                dtype=np.uint8
            )

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for b_idx, b in enumerate(x):
                size = b[3:] - b[:3]
                size[size < 0.1] = 0.1
                ori = b[:3]
                self.draw_cuboid(ax, ori, size, base_color[b_idx % len(base_color)] / 255)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_zlim(-1.5, 1.5)
            save_root = os.path.join(self.logger.log_dir, 'image')
            os.makedirs(save_root, exist_ok=True)
            plt.savefig(os.path.join(save_root, f'{self.global_step}_{batch_i}_pred_{len(x)}.png'), dpi=400)
            plt.close()
    
    def draw_cuboid(self, ax, origin, size, color):
        # Unpack the origin and size
        x, y, z = origin
        dx, dy, dz = size

        # Define vertices of the cuboid
        vertices = np.array([[x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
                             [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]])

        # Define the six faces of the cuboid
        faces = [[vertices[j] for j in [0, 1, 2, 3]],
                 [vertices[j] for j in [4, 5, 6, 7]], 
                 [vertices[j] for j in [0, 3, 7, 4]], 
                 [vertices[j] for j in [1, 2, 6, 5]], 
                 [vertices[j] for j in [0, 1, 5, 4]], 
                 [vertices[j] for j in [2, 3, 7, 6]]]

        # Create a 3D polygon collection and add to axis
        ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors=color, alpha=0.0))

        return


    def on_train_epoch_end(self):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', cur_lr, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config['lr'],
            weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.config['lr_decay'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='bbox/train.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    network = BBoxNet(config['network_params'])

    experiment = Trainer(config['trainer_params'], network)
    if config['trainer_params']['pretrained_model_path'] is not None:
        experiment.load_state_dict(
            torch.load(config['trainer_params']['pretrained_model_path'], 
                       map_location='cpu')['state_dict'], strict=True)

    dataset = Dataset(config['data_params'])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config['trainer_params']['batch_size'], 
        shuffle=True, num_workers=16)
    
    val_dataset = Dataset(config['data_params'])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, 
        shuffle=True, num_workers=16)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1, monitor='loss', mode='min',
        save_last=True, filename='{epoch}-{loss:.2f}')

    checkpoint_callback_last = pl.callbacks.ModelCheckpoint(
        filename='last-model-{epoch:02d}',
        save_last=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor()

    trainer_config = config['trainer_params']
    trainer = pl.Trainer(
        accelerator=trainer_config['accelerator'],
        max_epochs=trainer_config['max_epochs'],
        num_nodes=1, devices=trainer_config['devices'],
        strategy=trainer_config['strategy'],
        log_every_n_steps=trainer_config['log_every_n_steps'],
        default_root_dir=trainer_config['default_root_dir'],
        callbacks=[checkpoint_callback, checkpoint_callback_last, lr_monitor],
        gradient_clip_val=trainer_config['gradient_clip_val'],
        num_sanity_val_steps=1,
        limit_val_batches=1,
    )

    if trainer.is_global_zero:
        os.makedirs(trainer.log_dir, exist_ok=True)
        os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

    trainer.fit(experiment, dataloader, val_dataloader)



