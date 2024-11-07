import glob
import torch
import pickle
import os
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        data_path = self.config['data_path']
        if data_path.endswith('.pkl'):
            self.data_id_list = pickle.load(open(data_path, 'rb'))['train']
        else:
            self.data_id_list = glob.glob(os.path.join(data_path, '*', '*.npz'))
        self.data_id_list = sorted(self.data_id_list)
        if self.config['data_size'] > 0:
            self.data_id_list = self.data_id_list[:self.config['data_size']]
    
    def __len__(self):
        if len(self.data_id_list) < 200000:
            return 200000
        return len(self.data_id_list)
    
    def __getitem__(self, idx):
        idx = idx % len(self.data_id_list)
        data_path = self.data_id_list[idx]
        data = np.load(data_path)

        bbox = data['face_bboxes'] # (N, 6) [c_x, c_y, c_z, s_x, s_y, s_z]
        
        corner0 = bbox[:, :3] - bbox[:, 3:] / 2
        corner1 = bbox[:, :3] + bbox[:, 3:] / 2

        bbox = np.concatenate([corner0, corner1], axis=1)

        max_bbox_num = self.config['max_bbox_num']

        bbox = torch.from_numpy(bbox).float()

        if len(bbox) > max_bbox_num:
            return self.__getitem__(torch.randint(0, len(bbox), (1,)).item())
        elif len(bbox) < max_bbox_num:
            # padding
            repeat_time = max_bbox_num // len(bbox)
            sep = max_bbox_num - len(bbox) * repeat_time
            a = torch.cat([bbox[:sep],]*(repeat_time+1), dim=0)
            b = torch.cat([bbox[sep:],]*(repeat_time), dim=0)
            bbox = torch.cat([a, b], dim=0)
        
        # shuffle bbox
        idx = torch.randperm(bbox.shape[0])
        bbox = bbox[idx]

        return bbox


if __name__ == "__main__":
    import yaml
    with open('bbox/train.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset = Dataset(config['data_params'])
    for i in range(len(dataset)):
        data = dataset[i]
        print(i)
        
        

