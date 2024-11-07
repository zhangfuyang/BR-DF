import glob
import torch
import pickle
import numpy as np
import os

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, data_config):
        self.data_config = data_config
        data_path = data_config['data_path']
        if data_path.endswith('.pkl'):
            self.data_list = pickle.load(open(data_path, 'rb'))['train']
        else:
            self.data_list = glob.glob(os.path.join(data_path, '*.pkl'))
        self.data_list = sorted(self.data_list)

        if data_config['data_size'] > 0:
            self.data_list = self.data_list[:data_config['data_size']]
    
    def __len__(self):
        if len(self.data_list) < 10000:
            return 1000000
        return len(self.data_list) * 10
    
    def __getitem__(self, idx):
        idx = idx % len(self.data_list)
        data_path = self.data_list[idx]
        try:
            pkl_path = data_path
            if not os.path.exists(pkl_path):
                raise Exception('Not found')
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        solid_voxel = data['voxel_sdf'] # 8,N,N,N
        face_voxel = data['face_bounded_distance_field'] # 8,N,N,N, M
        face_voxel = np.transpose(face_voxel, (4, 0, 1, 2, 3)) # M, 8, N,N,N
        face_bboxes = data['face_bboxes'] # M, 6 [c_x, c_y, c_z, s_x, s_y, s_z]
        
        corner0 = face_bboxes[:, :3] - face_bboxes[:, 3:] / 2
        corner1 = face_bboxes[:, :3] + face_bboxes[:, 3:] / 2
        bboxes = np.concatenate([corner0, corner1], axis=1)
        # add gaussian noise
        bboxes += np.random.normal(0, 0.01, bboxes.shape)
        
        num_faces = face_voxel.shape[0]
        if num_faces > self.data_config['max_data_faces']:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        
        if self.data_config['face_shuffle']:
            random_idx = torch.randperm(num_faces)
            face_voxel = face_voxel[random_idx]
            bboxes = bboxes[random_idx]
        
        solid_voxel = torch.from_numpy(solid_voxel).float()[None]
        face_voxel = torch.from_numpy(face_voxel).float()
        bboxes = torch.from_numpy(bboxes).float()

        return {'solid_voxel': solid_voxel, 'face_voxel': face_voxel, 
                'face_bboxes': bboxes,
                'filename': pkl_path.split('/')[-1].split('.')[0]}


if __name__ == "__main__":
    import yaml
    with open('bbox_sdf_diffusion/train.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_dataset = LatentDataset(config['data_params'])
    for idx in range(len(train_dataset)):
        data = train_dataset[idx]
    print('done')


