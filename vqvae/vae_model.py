"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.K = codebook_size
        self.D = embedding_dim
        self.embedding = nn.Embedding(self.K, self.D)
        #self.embedding.weight.data.uniform_(-1/self.K, 1/self.K)
        return
    
    def forward(self, latents, vq_weight):
        latents = latents.permute(0, 2, 3, 4, 1).contiguous() # (B, C, H, W, D) -> (B, H, W, D, C)
        latents_shape = latents.shape
        assert latents_shape[-1] == self.D
        flat_latents = latents.view(-1, self.D) # (B*H*W*D, C)

        # compute L2 distance between latents and embedding vectors
        dist = torch.sum(flat_latents**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - \
                2 * torch.matmul(flat_latents, self.embedding.weight.t())
        
        # get the encoding that has the min distance
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1) # (B*H*W*D, 1)

        # convert encoding indices into one-hot vectors
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_indices.shape[0], self.K).to(device)
        encoding_one_hot.scatter_(1, encoding_indices, 1) # (B*H*W*D, K)

        # quantize latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized_latents = quantized_latents.view(latents_shape) # (B, H, W, D, C)

        # compute loss for embedding
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * vq_weight + embedding_loss

        # add the residue back to latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 4, 1, 2, 3).contiguous(), vq_loss
    
    def get_codebook_indices(self, latents):
        latents = latents.permute(0, 2, 3, 4, 1).contiguous() # (B, C, H, W, D) -> (B, H, W, D, C)
        latents_shape = latents.shape
        assert latents_shape[-1] == self.D
        flat_latents = latents.view(-1, self.D) # (B*H*W*D, C)

        # compute L2 distance between latents and embedding vectors
        dist = torch.sum(flat_latents**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - \
                2 * torch.matmul(flat_latents, self.embedding.weight.t())
        
        # get the encoding that has the min distance
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1) # (B*H*W*D, 1)

        encoding_indices = encoding_indices.view(latents_shape[:-1]+(1,)) # (B, H, W, D, 1)

        return encoding_indices[...,0]

    def reset_codebook(self, reset_idx):
        self.embedding.weight.data[reset_idx] = torch.randn_like(self.embedding.weight.data[reset_idx])
        # sync the weight to all gpus
        torch.distributed.broadcast(self.embedding.weight.data, 0)
        torch.distributed.barrier()
        return


class VQVAE3D(nn.Module):
    def __init__(self, n_channels, with_conv=True, num_res_blocks=2, ch=64,
                 channels=(2, 4, 8, 8), voxel_size=64, **kwargs):
        super(VQVAE3D, self).__init__()
        _ch = ch
        _channels = tuple(channels)
        self.n_channels = n_channels
        self.ch = ch
        self.channels = [int(c) for c in _channels]
        self.with_conv = with_conv
        self.num_res_blocks = num_res_blocks
        self.voxel_size = voxel_size
        self.codebook_size = kwargs['codebook_size']
        self.codebook_dim = kwargs['codebook_dim']

        self.vq_layer = VectorQuantizer(self.codebook_size, self.codebook_dim)

        self.conv_in = nn.Conv3d(n_channels, _ch, 
                                kernel_size=3, stride=1, padding=1)

        self.encoder = nn.ModuleList()
        in_channels = (1, ) + _channels
        for i in range(len(_channels)):
            if i != len(_channels)-1:
                self.encoder.append(
                    Down(in_channels[i]*_ch, in_channels[i+1]*_ch, 
                         num_res_blocks=self.num_res_blocks, with_conv=self.with_conv)
                )
            else:
                self.encoder.append(
                    Mid(in_channels[i]*_ch, in_channels[i+1]*_ch, 
                        num_res_blocks=self.num_res_blocks)
                )
        self.encoder.append(
            Mid(in_channels[-1]*_ch, self.codebook_dim, num_res_blocks=self.num_res_blocks)
        )
            
        self.decoder_input = Mid(self.codebook_dim, in_channels[-1]*_ch, self.num_res_blocks)
        self.decoder = nn.ModuleList()
        for i in range(len(_channels), 0, -1):
            if i == len(_channels):
                self.decoder.append(
                    Mid(in_channels[i]*_ch, in_channels[i-1]*_ch, 
                        num_res_blocks=self.num_res_blocks)
                )
            else:
                self.decoder.append(
                    Up(in_channels[i]*_ch, in_channels[i-1]*_ch, 
                       num_res_blocks=self.num_res_blocks, with_conv=self.with_conv)
                )

        self.final_layer = nn.Sequential(
            nn.Conv3d(in_channels[0]*_ch, n_channels, kernel_size=1),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.conv_in(x)
        for model in self.encoder:
            x = model(x)

        return x
    
    def decode(self, z):
        z = self.decoder_input(z)
        for model in self.decoder:
            z = model(z)
        result = self.final_layer(z)
        return result
    
    def forward(self, x, vq_weight):
        latent = self.encode(x)
        quantized_inputs, vq_loss = self.vq_layer(latent, vq_weight)
        code_indices = self.vq_layer.get_codebook_indices(latent)
        return self.decode(quantized_inputs), vq_loss, code_indices
    
    def quantize_decode(self, latent):
        quantized_inputs, _ = self.vq_layer(latent, 0)
        return self.decode(quantized_inputs)

    def quantize_indices(self, x):
        latent = self.encode(x)
        return self.vq_layer.get_codebook_indices(latent)

    def loss_function(self, recon_x, x, vq_loss, **kwargs):
        recons_loss = F.mse_loss(recon_x, x)
        #if kwargs['recon_loss_type'] == 'l1':
        #    diff = torch.abs(recon_x - x)
        #elif kwargs['recon_loss_type'] == 'l2':
        #    diff = (recon_x - x)**2
        #weight = 1 - torch.abs(x)
        #recons_loss = torch.mean(diff * weight)

        loss = recons_loss + vq_loss

        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'VQ_Loss':vq_loss}
    
    def reset_codebook(self, reset_idx):
        self.vq_layer.reset_codebook(reset_idx)    


class Mid(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResNetBlock(in_channels, out_channels))
            in_channels = out_channels
        
    def forward(self, x):
        for model in self.res_blocks:
            x = model(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2, with_conv=True):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResNetBlock(in_channels, out_channels))
            in_channels = out_channels
        
        self.with_conv = with_conv
        if with_conv:
            self.downsample = nn.Conv3d(in_channels, out_channels, 
                                    kernel_size=3, stride=2, padding=0)
        else:
            self.downsample = nn.AvgPool3d(2)

    def forward(self, x):
        for model in self.res_blocks:
            x = model(x)
        if self.with_conv:
            pad = (0,1,0,1,0,1)
            x = F.pad(x, pad, "constant", 0)
        return self.downsample(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2, with_conv=True):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResNetBlock(in_channels, out_channels))
            in_channels = out_channels
        
        self.with_conv = with_conv
        if with_conv:
            self.upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        for model in self.res_blocks:
            x = model(x)
        return self.upsample(x)

def Normalize(in_channels, num_groups=32):
    #return torch.nn.BatchNorm3d(num_features=in_channels)
    return torch.nn.InstanceNorm3d(num_features=in_channels)
    #return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1)
        
        self.nonlinearity = nn.SiLU()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h

