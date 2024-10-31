import glob
import torch
import pickle
import numpy as np
import os

class VoxelDataset(torch.utils.data.Dataset):
    def __init__(self, data_config):
        self.data_config = data_config
        if self.data_config['data_path'].endswith('.pkl'):
            self.data_list = pickle.load(open(data_config['data_path'], 'rb'))
        elif self.data_config['data_path'].endswith('.txt'):
            self.data_list = []
            with open(data_config['data_path'], 'r') as f:
                for line in f:
                    self.data_list.append(line.strip())
        else:
            data_format = self.data_config['data_format']
            self.data_list = glob.glob(os.path.join(data_config['data_path'], '*', f'*.{data_format}'))
            self.data_list = sorted(self.data_list)
        
        if self.data_config['filter_data_path'] is not None:
            filter_list = pickle.load(open(self.data_config['filter_data_path'], 'rb'))
            filter_list = [x.split('/')[-2]+'_'+x.split('/')[-1].split('.')[0] for x in filter_list]
            new_data_list = []
            for data in self.data_list:
                data_name = data.split('/')[-2]+'_'+data.split('/')[-1].split('.')[0]
                if data_name not in filter_list:
                    new_data_list.append(data)
            self.data_list = new_data_list

    def __len__(self):
        return len(self.data_list) 
    
    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        try:
            if data_path.endswith('.npz'):
                data = np.load(data_path)
            elif data_path.endswith('.pkl'):
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
        except:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        voxel = data[self.data_config['data_key']]
        if voxel.ndim == 4:
            random_idx = torch.randint(0, voxel.shape[-1], (1,)).item()
            voxel = voxel[..., random_idx]
        
        x = torch.from_numpy(voxel).float()
        x = torch.clamp(x, -self.data_config['clip_value'], self.data_config['clip_value'])
        x = x * self.data_config['scale_factor']
        x = x[None] # 1, N,N,N
        return x, data_path


class ReconDataset(torch.utils.data.Dataset):
    def __init__(self, data_config):
        self.data_config = data_config
        if self.data_config['data_path'].endswith('.pkl'):
            self.data_list = pickle.load(open(data_config['data_path'], 'rb'))['train']
        elif self.data_config['data_path'].endswith('.txt'):
            self.data_list = []
            with open(data_config['data_path'], 'r') as f:
                for line in f:
                    self.data_list.append(line.strip())
        else:
            data_format = self.data_config['data_format']
            self.data_list = glob.glob(os.path.join(data_config['data_path'], '*', f'*.{data_format}'))
            self.data_list = sorted(self.data_list)

        if self.data_config['filter_data_path'] is not None:
            filter_list = pickle.load(open(self.data_config['filter_data_path'], 'rb'))
            filter_list = [x.split('/')[-2]+'_'+x.split('/')[-1].split('.')[0] for x in filter_list]
            new_data_list = []
            for data in self.data_list:
                data_name = data.split('/')[-2]+'_'+data.split('/')[-1].split('.')[0]
                if data_name not in filter_list:
                    new_data_list.append(data)
                else:
                    print('filter', data_name)
            self.data_list = new_data_list
        
        temp_list = []
        for data in self.data_list:
            temp_list.append(data.replace('data', 'data2').replace('solid', 'solid_new'))
        self.data_list = temp_list

        self.augmentation = self.data_config['augmentation']
        self.aug_list = []
        # identity augmentation
        self.aug_list.append('identity')
        if self.augmentation:
            # flip augmentation
            self.aug_list.append('flip-x')
            self.aug_list.append('flip-y')
            self.aug_list.append('flip-z')

            # rotation augmentation
            self.aug_list.append('rotate-x-90')
            self.aug_list.append('rotate-y-90')
            self.aug_list.append('rotate-z-90')
            self.aug_list.append('rotate-x-180')
            self.aug_list.append('rotate-y-180')
            self.aug_list.append('rotate-z-180')
            self.aug_list.append('rotate-x-270')
            self.aug_list.append('rotate-y-270')
            self.aug_list.append('rotate-z-270')

            # swap augmentation
            self.aug_list.append('swap-x-y')
            self.aug_list.append('swap-x-z')
            self.aug_list.append('swap-y-z')

    def __len__(self):
        return len(self.data_list) * len(self.aug_list)
    
    def augment(self, solid_voxel, faces_voxel, aug_type):
        # solid_voxel: 1, N,N,N
        # faces_voxel: M, N,N,N
        if aug_type == 'identity':
            return solid_voxel, faces_voxel
        elif aug_type == 'flip-x':
            return solid_voxel.flip(1), faces_voxel.flip(1)
        elif aug_type == 'flip-y':
            return solid_voxel.flip(2), faces_voxel.flip(2)
        elif aug_type == 'flip-z':
            return solid_voxel.flip(3), faces_voxel.flip(3)
        elif aug_type == 'rotate-x-90':
            return solid_voxel.rot90(1, (2,3)), faces_voxel.rot90(1, (2,3))
        elif aug_type == 'rotate-y-90':
            return solid_voxel.rot90(1, (1,3)), faces_voxel.rot90(1, (1,3))
        elif aug_type == 'rotate-z-90':
            return solid_voxel.rot90(1, (1,2)), faces_voxel.rot90(1, (1,2))
        elif aug_type == 'rotate-x-180':
            return solid_voxel.rot90(2, (2,3)), faces_voxel.rot90(2, (2,3))
        elif aug_type == 'rotate-y-180':
            return solid_voxel.rot90(2, (1,3)), faces_voxel.rot90(2, (1,3))
        elif aug_type == 'rotate-z-180':
            return solid_voxel.rot90(2, (1,2)), faces_voxel.rot90(2, (1,2))
        elif aug_type == 'rotate-x-270':
            return solid_voxel.rot90(3, (2,3)), faces_voxel.rot90(3, (2,3))
        elif aug_type == 'rotate-y-270':
            return solid_voxel.rot90(3, (1,3)), faces_voxel.rot90(3, (1,3))
        elif aug_type == 'rotate-z-270':
            return solid_voxel.rot90(3, (1,2)), faces_voxel.rot90(3, (1,2))
        elif aug_type == 'swap-x-y':
            return solid_voxel.permute(0,2,1,3), faces_voxel.permute(0,2,1,3)
        elif aug_type == 'swap-x-z':
            return solid_voxel.permute(0,3,2,1), faces_voxel.permute(0,3,2,1)
        elif aug_type == 'swap-y-z':
            return solid_voxel.permute(0,1,3,2), faces_voxel.permute(0,1,3,2)
        else:
            raise ValueError(f'Augmentation type {aug_type} not supported')

    def __getitem__(self, idx):
        data_idx = idx // len(self.aug_list)
        aug_idx = idx % len(self.aug_list)
        data_path = self.data_list[data_idx]
        if data_path.endswith('.npz'):
            data = np.load(data_path)
        elif data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        solid_voxel = data['voxel_sdf'] # N,N,N
        faces_voxel = data['faces_udf_reg'] # N,N,N,M

        solid_voxel = torch.from_numpy(solid_voxel).float()
        faces_voxel = torch.from_numpy(faces_voxel).float()
        solid_voxel = torch.clamp(solid_voxel, -self.data_config['clip_value'], self.data_config['clip_value'])
        faces_voxel = torch.clamp(faces_voxel, -self.data_config['clip_value'], self.data_config['clip_value'])
        solid_voxel = solid_voxel * self.data_config['scale_factor']
        faces_voxel = faces_voxel * self.data_config['scale_factor']

        solid_voxel = solid_voxel.unsqueeze(0) # 1, N,N,N
        faces_voxel = faces_voxel.permute(3, 0, 1, 2) # M, N,N,N

        aug_type = self.aug_list[aug_idx]
        solid_voxel, faces_voxel = self.augment(solid_voxel, faces_voxel, aug_type)

        data_name = data_path.split('/')[-2]+'_'+data_path.split('/')[-1].split('.')[0]+'_'+aug_type

        face_bboxes = data['face_bboxes']
        face_bboxes = torch.from_numpy(face_bboxes).float()

        return solid_voxel, faces_voxel, face_bboxes, data_name


if __name__ == "__main__":
    import yaml
    import mcubes
    with open('vqvae/configs/recon.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_dataset = ReconDataset(config['data_params']['train'])
    print(len(train_dataset))
    for idx, data in enumerate(train_dataset):
        print()
    
    # save bad list
    with open('Data/processed/deepcad_subset/train_bad_list.pkl', 'wb') as f:
        pickle.dump(bad_list, f)

