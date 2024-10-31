import os
import torch
import pytorch_lightning as pl
import mcubes
import numpy as np
from torch.optim.optimizer import Optimizer
import trimesh
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.utils import BaseOutput
import pickle
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator
import math
from utils import draw_cuboid
import matplotlib.pyplot as plt

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
class DiffusionExperiment(pl.LightningModule):
    def __init__(self, config, diffusion_model, face_model, sdf_model):
        super(DiffusionExperiment, self).__init__()
        self.config = config
        self.face_model = face_model
        self.sdf_model = sdf_model
        if self.face_model is not None:
            self.face_model.eval()
            for param in self.face_model.parameters():
                param.requires_grad = False
        if self.sdf_model is not None:
            self.sdf_model.eval()
            for param in self.sdf_model.parameters():
                param.requires_grad = False
        self.diffusion_model = diffusion_model
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.02,
            beta_schedule='scaled_linear',
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type=self.config['prediction_type'],
            )
        
        self.latent_dim = None
        self.test_idx = 0

        # load mean and std
        if config['latent_std_mean_path'] is not None:
            with open(config['latent_std_mean_path'], 'rb') as f:
                data = pickle.load(f)
            self.solid_latent_mean = data['solid_mean']
            self.solid_latent_std = data['solid_std']
            if self.config['visual_type'] == 'sanity_check':
                self.face_latent_mean = self.solid_latent_mean
                self.face_latent_std = self.solid_latent_std
            else:
                self.face_latent_mean = data['face_mean']
                self.face_latent_std = data['face_std']
        else:
            self.face_latent_mean = 0
            self.face_latent_std = 1
            self.solid_latent_mean = 0
            self.solid_latent_std = 1

    @torch.no_grad()
    def preprocess(self, batch):
        solid_voxel = batch['solid_voxel'] # bs, 1, N, N, N or bs, 1, C, N, N, N
        faces_voxel = batch['faces_voxel'] # bs, M, N, N, N or bs, M, C, N, N, N

        # get latent
        if solid_voxel.dim() == 5:
            # voxel
            with torch.no_grad():
                solid_latent = self.sdf_model.encode(solid_voxel)[:,None] # bs, 1, C, N, N, N

                self.latent_dim = solid_latent.shape[2]

                bs = face_voxel.shape[0]
                face_voxel = face_voxel.reshape(-1, *face_voxel.shape[2:])[:,None]
                face_latent = self.face_model.encode(face_voxel) # bs*M, C, N, N, N
                face_latent = face_latent.reshape(bs, -1, *face_latent.shape[1:])
                # bs, M, C, N, N, N
        else:
            self.latent_dim = solid_voxel.shape[2]
            solid_latent = solid_voxel
            face_latent = faces_voxel
        
        # normalize
        solid_latent = (solid_latent - self.solid_latent_mean) / self.solid_latent_std
        face_latent = (face_latent - self.face_latent_mean) / self.face_latent_std

        bs = solid_latent.shape[0]
        latent = torch.cat([solid_latent, face_latent], 1) # bs, 1+M, C, N, N, N

        face_bbox = batch['faces_bbox'] # bs, M, 6
        mask = batch['mask'] # bs, M

        return solid_latent, face_latent, face_bbox, mask
    
    def training_step(self, batch, batch_idx):
        solid_latent, face_latent, face_bbox, mask = self.preprocess(batch)
        
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (solid_latent.shape[0],), device=solid_latent.device).long()
        noise_face = torch.randn_like(face_latent)
        noise_solid = torch.randn_like(solid_latent)
        z_faces = self.scheduler.add_noise(face_latent, noise_face, t)
        z_solid = self.scheduler.add_noise(solid_latent, noise_solid, t)
        
        model_output_solid, model_output_faces = self.diffusion_model(
            z_solid, z_faces, face_bbox, mask, t)

        solid_target = solid_latent if self.config['prediction_type'] == 'sample' else noise_solid
        face_target = face_latent if self.config['prediction_type'] == 'sample' else noise_face
        
        target = torch.cat([solid_target, face_target], 1)
        model_output = torch.cat([model_output_solid, model_output_faces], 1)
        if self.config['loss_type'] == 'l2':
            sd_loss = torch.nn.functional.mse_loss(model_output, target)
        elif self.config['loss_type'] == 'l1':
            sd_loss = torch.nn.functional.l1_loss(model_output, target)

        self.log('train_loss', sd_loss, rank_zero_only=True, prog_bar=True)

        loss = sd_loss 

        return loss
    
    def on_train_epoch_end(self) -> None:
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        solid_latent, face_latent, face_bbox, mask = self.preprocess(batch)
        z_faces = torch.randn_like(face_latent)
        z_solid = torch.randn_like(solid_latent)

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = torch.cat([t.unsqueeze(0)]*z_faces.shape[0], 0)
            model_output_solid, model_output_faces = self.diffusion_model(
                z_solid, z_faces, face_bbox, mask, timestep)
            z_solid = self.scheduler.step(model_output_solid, t, z_solid).prev_sample
            z_faces = self.scheduler.step(model_output_faces, t, z_faces).prev_sample

        if self.trainer.is_global_zero:
            if batch_idx == 0: 
                for i in range(min(40, solid_latent.shape[0])):
                    face_num = batch['faces_num'][i]
                    if self.config['visual_type'] == 'vanilla':
                        os.makedirs(os.path.join(self.logger.log_dir, 'image'), exist_ok=True)

                        # render bbox
                        bbox = face_bbox[i][:face_num].cpu().numpy() # M, 6
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        for b_idx, b in enumerate(bbox):
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
                        plt.savefig(os.path.join(self.logger.log_dir, 'image', f'{self.global_step}_{i}_bbox.png'), dpi=300)
                        plt.close()

                        save_name_prefix = os.path.join(self.logger.log_dir, 'image', f'{self.global_step}_{i}_pred')
                        solid_voxel, face_voxel = self.latent_to_voxel(z_solid[i][0], z_faces[i, :face_num])
                        
                        solid_mesh = self.render_mesh(solid_voxel, phase='sdf')
                        solid_mesh.export(save_name_prefix+'_solid.obj')

                        face_meshes = None
                        for face_idx in range(face_voxel.shape[0]):
                            face_mesh = self.render_mesh(face_voxel[face_idx], phase='face')
                            if face_mesh.vertices.shape[0] == 0:
                                continue
                            face_mesh.visual.vertex_colors = base_color[face_idx % len(base_color)]
                            if face_meshes is None:
                                face_meshes = face_mesh
                            else:
                                face_meshes += face_mesh
                        face_meshes.export(save_name_prefix+f'_face_{face_voxel.shape[0]}.obj', include_color=True)

                        save_name_prefix = os.path.join(self.logger.log_dir, 'image', f'{self.global_step}_{i}_gt')
                        solid_voxel, face_voxel = self.latent_to_voxel(solid_latent[i][0], face_latent[i, :face_num])
                        solid_mesh = self.render_mesh(solid_voxel, phase='sdf')
                        solid_mesh.export(save_name_prefix+'_solid.obj')
                        face_meshes = None
                        for face_idx in range(face_voxel.shape[0]):
                            face_mesh = self.render_mesh(face_voxel[face_idx], phase='face')
                            if face_mesh.vertices.shape[0] == 0:
                                continue
                            face_mesh.visual.vertex_colors = base_color[face_idx % len(base_color)]
                            if face_meshes is None:
                                face_meshes = face_mesh
                            else:
                                face_meshes += face_mesh
                        face_meshes.export(save_name_prefix+'_face.obj', include_color=True)
                    elif self.config['visual_type'] == 'brep':
                        save_name = os.path.join(self.logger.log_dir, 'val', f'{self.global_step}', f'{i}_pred', 'raw.pkl')
                        self.save_result(z_solid[i][0], z_faces[i], save_name)
                        save_name = os.path.join(self.logger.log_dir, 'val', f'{self.global_step}', f'{i}_gt', 'raw.pkl')
                        self.save_result(solid_latent[i][0], face_latent[i], save_name)
                        # make a file with filename
                        gt_filename = batch['filename'][i]
                        with open(os.path.join(os.path.dirname(save_name), f'{gt_filename}'), 'w') as f:
                            f.write(gt_filename)

    def validation_step_2(self, batch, batch_idx):
        repaint_scheduler = RepaintDDIM(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.02,
            beta_schedule='scaled_linear',
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type=self.config['prediction_type'],
            )
        x = self.preprocess(batch)
        face_latent = x[:,1:]
        solid_latent = x[:,:1]
        z_faces = torch.randn_like(face_latent)
        z_solid = torch.randn_like(solid_latent)

        repaint_scheduler.set_timesteps(self.config['diffusion_steps'], jump_length=10, jump_n_sample=10, device=x.device)
        timesteps = repaint_scheduler.timesteps

        t_last = repaint_scheduler.timesteps[0] + 1
        mask = torch.ones_like(x)
        #mask[:,1:] = 0

        mask[:,1:,:,   :, :,] = 0
        
        for i, t in enumerate(timesteps):
            z_solid_input = z_solid 
            z_faces_input = z_faces
            z = torch.cat([z_solid, z_faces], 1)
            if t < t_last:
                timestep = torch.cat([t.unsqueeze(0)]*z_faces_input.shape[0], 0)
                model_output_face, model_output_solid = self.diffusion_model(
                                    z_faces_input, z_solid_input, timestep)
                model_output = torch.cat([model_output_solid, model_output_face], 1)
                z = repaint_scheduler.step(model_output, t, z, x, mask).prev_sample
            else:
                z = repaint_scheduler.undo_step(z, t_last)
            z_solid = z[:,:1]
            z_faces = z[:,1:]
            t_last = t
            
        if self.trainer.is_global_zero:
            if batch_idx == 0: 
                for i in range(min(40, x.shape[0])):
                    if self.config['visual_type'] == 'vanilla':
                        os.makedirs(os.path.join(self.logger.log_dir, 'image'), exist_ok=True)
                        save_name_prefix = os.path.join(self.logger.log_dir, 'image', f'{self.global_step}_{i}_pred')
                        solid_voxel, face_voxel = self.latent_to_voxel(z_solid[i][0], z_faces[i])
                        
                        solid_mesh = self.render_mesh(solid_voxel, phase='sdf')
                        solid_mesh.export(save_name_prefix+'_solid.obj')

                        face_meshes = None
                        face_voxel = self.nms(face_voxel)
                        for face_idx in range(min(face_voxel.shape[0], base_color.shape[0])):
                            face_mesh = self.render_mesh(face_voxel[face_idx], phase='face')
                            if face_mesh.vertices.shape[0] == 0:
                                continue
                            face_mesh.visual.vertex_colors = base_color[face_idx]
                            if face_meshes is None:
                                face_meshes = face_mesh
                            else:
                                face_meshes += face_mesh
                        face_meshes.export(save_name_prefix+f'_face_{face_voxel.shape[0]}.obj', include_color=True)

                        
                        save_name_prefix = os.path.join(self.logger.log_dir, 'image', f'{self.global_step}_{i}_pred_latent_nms')
                        solid_voxel, face_voxel = self.latent_to_voxel(z_solid[i][0], self.nms_latent(z_faces[i]))
                        face_meshes = None
                        for face_idx in range(min(face_voxel.shape[0], base_color.shape[0])):
                            face_mesh = self.render_mesh(face_voxel[face_idx], phase='face')
                            if face_mesh.vertices.shape[0] == 0:
                                continue
                            face_mesh.visual.vertex_colors = base_color[face_idx]
                            if face_meshes is None:
                                face_meshes = face_mesh
                            else:
                                face_meshes += face_mesh
                        face_meshes.export(save_name_prefix+f'_face_{face_voxel.shape[0]}.obj', include_color=True)

                        
                        save_name_prefix = os.path.join(self.logger.log_dir, 'image', f'{self.global_step}_{i}_gt')
                        solid_voxel, face_voxel = self.latent_to_voxel(solid_latent[i][0], face_latent[i])
                        solid_mesh = self.render_mesh(solid_voxel, phase='sdf')
                        solid_mesh.export(save_name_prefix+'_solid.obj')
                        face_meshes = None
                        face_voxel = self.nms(face_voxel)
                        for face_idx in range(min(face_voxel.shape[0], base_color.shape[0])):
                            face_mesh = self.render_mesh(face_voxel[face_idx], phase='face')
                            if face_mesh.vertices.shape[0] == 0:
                                continue
                            face_mesh.visual.vertex_colors = base_color[face_idx]
                            if face_meshes is None:
                                face_meshes = face_mesh
                            else:
                                face_meshes += face_mesh
                        face_meshes.export(save_name_prefix+'_face.obj', include_color=True)
                    elif self.config['visual_type'] == 'brep':
                        save_name = os.path.join(self.logger.log_dir, 'val', f'{self.global_step}', f'{i}_pred', 'raw.pkl')
                        self.save_result(z_solid[i][0], z_faces[i], save_name)
                        save_name = os.path.join(self.logger.log_dir, 'val', f'{self.global_step}', f'{i}_gt', 'raw.pkl')
                        self.save_result(solid_latent[i][0], face_latent[i], save_name)
                        # make a file with filename
                        gt_filename = batch['filename'][i]
                        with open(os.path.join(os.path.dirname(save_name), f'{gt_filename}'), 'w') as f:
                            f.write(gt_filename)

    def latent_to_voxel(self, sdf_latent, face_latents):
        if sdf_latent is not None:
            sdf_latent = sdf_latent[None] # 1, C, N, N, N
            sdf_latent = sdf_latent * self.solid_latent_std + self.solid_latent_mean
            with torch.no_grad():
                sdf_voxel = self.sdf_model.quantize_decode(sdf_latent) # 1, 1, N, N, N
            sdf_voxel = sdf_voxel[0,0]
        else:
            sdf_voxel = None

        if face_latents is not None:
            face_latents = face_latents * self.face_latent_std + self.face_latent_mean
            with torch.no_grad():
                face_voxel = self.face_model.quantize_decode(face_latents) # M, 1, N, N, N
            face_voxel = face_voxel[:,0]
        else:
            face_voxel = None
        
        return sdf_voxel, face_voxel

    def render_mesh(self, voxel, filename=None, phase='sdf'):
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        voxel = voxel.cpu().numpy()
        if phase == 'sdf':
            vertices, triangles = mcubes.marching_cubes(voxel, 0)
            if filename is None:
                return trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
            mcubes.export_obj(vertices, triangles, filename)
        elif phase == 'face':
            points = np.where(voxel < 0.2)
            points = np.array(points).T
            pointcloud = trimesh.points.PointCloud(points)
            if filename is None:
                return pointcloud
            # save
            pointcloud.export(filename)
        else:
            raise ValueError(f'phase {phase} not supported')

    def render_mesh_2(self, solid_voxel, face_voxel, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        solid_voxel = solid_voxel.cpu().numpy()
        face_voxel = face_voxel.cpu().numpy()
        face_voxel = face_voxel.transpose(1, 2, 3, 0) # N, N, N, M
        vertices, triangles = mcubes.marching_cubes(solid_voxel, 0)
        grid_reso = face_voxel.shape[0]
        f_dbf_interpolator = RegularGridInterpolator(
            (np.arange(grid_reso), np.arange(grid_reso), np.arange(grid_reso)), face_voxel, 
            bounds_error=False, fill_value=0)
        interpolated_f_bdf = f_dbf_interpolator(vertices) # v, num_faces
        vertices_face_id = interpolated_f_bdf.argmin(-1)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        if vertices_face_id.shape[0] != 0:
            mesh.visual.vertex_colors = base_color[vertices_face_id % len(base_color)]
        mesh.export(filename, include_color=True)

    def configure_optimizers(self):
        for module_name in self.config['freeze_modules']:
            module = self.diffusion_model.__dict__['_modules'][module_name]
            for param in module.parameters():
                param.requires_grad = False
        p = filter(lambda p: p.requires_grad, self.diffusion_model.parameters())
        optimizer = torch.optim.Adam(
            p, lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_decay'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def save_result(self, solid_latent, face_latent, save_name):
        solid_voxel, face_voxels = self.latent_to_voxel(solid_latent, face_latent)
        #face_voxels = self.nms(face_voxels)

        #self.render_mesh_2(solid_voxel, face_voxels, os.path.dirname(save_name)+f'/mc.obj')
        solid_voxel = solid_voxel.cpu().numpy()
        face_voxels = face_voxels.cpu().numpy()

        solid_voxel = solid_voxel / 10.
        face_voxels = face_voxels / 10.

        face_voxels = face_voxels.transpose(1, 2, 3, 0)

        data = {}
        data['voxel_sdf'] = solid_voxel
        data['face_bounded_distance_field'] = face_voxels
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        with open(save_name, 'wb') as f:
            pickle.dump(data, f)
        command = f'python brep_render.py --data_path {save_name} --save_root {os.path.join(os.path.dirname(save_name), "processed")} ' \
                    f'--apply_nms --vis_each_face --vis_face_all --vis_face_only --vis_each_boundary'
        print(command)
        os.system(command)

        # save raw
        os.makedirs(os.path.join(os.path.dirname(save_name), 'raw'), exist_ok=True)
        for face_i in range(face_voxels.shape[-1]):
            face_v = face_voxels[:,:,:,face_i]
            points = np.where(face_v < 0.02)
            points = np.array(points).T
            pointcloud = trimesh.points.PointCloud(points)
            # save
            pointcloud.export(os.path.join(os.path.dirname(save_name), 'raw', f'face_{face_i}_only.obj'))

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        x = self.preprocess(batch)
        face_latent = x[:,1:]
        solid_latent = x[:,:1]
        z_faces = torch.randn_like(face_latent)
        z_solid = torch.randn_like(solid_latent)

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            z_faces_input = z_faces 
            z_solid_input = z_solid 
            timestep = torch.cat([t.unsqueeze(0)]*z_faces_input.shape[0], 0)
            model_output_face, model_output_solid = self.diffusion_model(
                                z_faces_input, z_solid_input, timestep)
            z_solid = self.scheduler.step(model_output_solid, t, z_solid).prev_sample
            z_faces = self.scheduler.step(model_output_face, t, z_faces).prev_sample

        if self.trainer.is_global_zero:
            for i in range(x.shape[0]):
                cur_test_idx = self.test_idx + i
                save_name = os.path.join(self.logger.log_dir, 'test', f'{cur_test_idx:04d}', 'pred', 'raw.pkl')
                self.save_result(z_solid[i][0], z_faces[i], save_name)
                save_name = os.path.join(self.logger.log_dir, 'test', f'{cur_test_idx:04d}', f'gt', 'raw.pkl')
                self.save_result(x[i][0], x[i][1:], save_name)

        self.test_idx += x.shape[0]
    
