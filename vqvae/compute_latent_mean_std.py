import pickle
import numpy as np
import os
import glob

data_dir = 'vqvae/logs/recon/train_npz_reso_4/lightning_logs/version_2/pkl'
save_path = 'vqvae/logs/recon/train_npz_reso_4/lightning_logs/version_2/mean_std.pkl'

voxel_all = []
face_all = []
for pkl_name in glob.glob(os.path.join(data_dir, '*.pkl')):
    with open(pkl_name, 'rb') as f:
        data = pickle.load(f)
    voxel_sdf = data['voxel_sdf'][..., None]
    face_udf = data['face_bounded_distance_field']

    voxel_all.append(voxel_sdf)
    face_all.append(face_udf)

voxel_sdf = np.concatenate(voxel_all, axis=-1)
face_udf = np.concatenate(face_all, axis=-1)

voxel_mean, voxel_std = np.mean(voxel_sdf), np.std(voxel_sdf)
face_mean, face_std = np.mean(face_udf), np.std(face_udf)

print(f'voxel mean: {voxel_mean}, voxel std: {voxel_std}')
print(f'face mean: {face_mean}, face std: {face_std}')
print(f'voxel max: {np.max(voxel_sdf)}, voxel min: {np.min(voxel_sdf)}')
print(f'face max: {np.max(face_udf)}, face min: {np.min(face_udf)}')

# save pkl
data = {'solid_mean': voxel_mean, 
        'solid_std': voxel_std, 
        'face_mean': face_mean, 
        'face_std': face_std}
with open(save_path, 'wb') as f:
    pickle.dump(data, f)

