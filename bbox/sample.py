import os
import torch
import argparse
import yaml
from network import BBoxNet
from diffusers import DDPMScheduler, PNDMScheduler
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='bbox/train.yaml')
    parser.add_argument('--checkpoint', type=str, default='bbox/logs/debug/lightning_logs/version_1/checkpoints/last.ckpt')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    network = BBoxNet(config['network_params'])
    # load checkpoint
    state_dict = torch.load(args.checkpoint, map_location='cpu')['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('network.'):
            new_state_dict[k[8:]] = v
        else:
            new_state_dict[k] = v
    network.load_state_dict(new_state_dict, strict=True)

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
        clip_sample_range=2,)
    

    network = network.to('cuda').eval()

    batch_size = 16
    num_bbox = 50
    bbox_threshold = 0.05
    with torch.no_grad():
        bbox = torch.randn(batch_size, num_bbox, 4).float().to('cuda')

        pndm_scheduler.set_timesteps(200)
        for t in tqdm(pndm_scheduler.timesteps[:158]):
            timesteps = torch.full((batch_size,), t, device='cuda').long()
            pred = network(bbox, timesteps)
            bbox = pndm_scheduler.step(pred, t, bbox).prev_sample
        
        ddpm_scheduler.set_timesteps(1000)
        for t in tqdm(ddpm_scheduler.timesteps[-250:]):
            timesteps = torch.full((batch_size,), t, device='cuda').long()
            pred = network(bbox, timesteps)
            bbox = ddpm_scheduler.step(pred, t, bbox).prev_sample
    
    # remove duplicate bbox
    bbox_raw = bbox.cpu().numpy() # (bs, num_bbox, 4) x,y,z,size
    bbox_corner1 = bbox_raw[:,:,:3]-bbox_raw[:,:,3:]/2
    bbox_corner2 = bbox_raw[:,:,:3]+bbox_raw[:,:,3:]/2
    bbox_check = np.concatenate([bbox_corner1[:,:,None], bbox_corner2[:,:,None]], axis=2) 
    # (bs, num_bbox, 2, 3) 

    bbox_deduplicate = []
    for b_i in range(batch_size):
        x = np.round(bbox_check[b_i], 4)
        non_repeat = x[:1]
        non_repeat_idx = [0]
        for i in range(1, len(x)):
            diff = np.max(np.max(np.abs(non_repeat - x[i]), -1), -1)
            same = diff < bbox_threshold
            if same.sum()>=1:
                continue
            non_repeat = np.concatenate([non_repeat, x[i][None]], axis=0)
            non_repeat_idx.append(i)
        
        deduplicate = bbox_raw[b_i][non_repeat_idx]
        print(deduplicate)




    # save bbox
