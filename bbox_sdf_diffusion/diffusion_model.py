import torch.nn as nn
import torch
from model_utils import Timesteps, TimestepEmbedding, \
        get_parameter_dtype, EncodeBlock3D, CrossAttnEncodeBlock3D, \
        EncodeDownBlock3D, DecodeBlock3D, DecodeUpBlock3D, ResNetBlockEmbed, \
        CrossEncodeBlock3D
        

class Solid3DModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        freq_shift = config['freq_shift']
        flip_sin_to_cos = config['flip_sin_to_cos']
        solid_params = config['solid_params']
        face_params = config['face_params']
        block_channels = config['block_channels']
        layers_per_block = config['layers_per_block']
        act_fn = config['act_fn']
        attention_head_dim = config['attention_head_dim']
        norm_num_groups = config['norm_num_groups']
        norm_eps = config['norm_eps']
        time_embed_dim = config['time_embed_dim']
        bbox_embed_dim = config['bbox_embed_dim']

        self.solid_model = UNet3DModel(in_channels, out_channels, freq_shift,
                                        flip_sin_to_cos, solid_params,
                                        block_channels, layers_per_block,
                                        act_fn, attention_head_dim, norm_num_groups,
                                        norm_eps, time_embed_dim, bbox_embed_dim)
        self.face_model = UNet3DModel(in_channels, out_channels, freq_shift, 
                                     flip_sin_to_cos, face_params, 
                                     block_channels, layers_per_block, 
                                     act_fn, attention_head_dim, norm_num_groups, 
                                     norm_eps, time_embed_dim, bbox_embed_dim)

        # time embedding
        time_embed_dim = config['time_embed_dim']
        self.time_proj = Timesteps(time_embed_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(time_embed_dim, time_embed_dim)
        
        # bbox embedding
        bbox_embed_dim = config['bbox_embed_dim']
        self.bbox_embed = nn.Sequential(
            nn.Linear(6, bbox_embed_dim),
            nn.LayerNorm(bbox_embed_dim),
            nn.SiLU(),
            nn.Linear(bbox_embed_dim, bbox_embed_dim),
        )

        self.f2f_model = nn.ModuleList()
        self.s2f_model = nn.ModuleList()
        self.f2s_model = nn.ModuleList()
        cross_zero_init = True
        attn_channels = []
        for block in self.solid_model.encode_blocks:
            attn_channels.append(block.resnets[0].conv1.in_channels)
        for block in self.solid_model.mid_block:
            attn_channels.append(block.resnets[0].conv1.in_channels)
        for block in self.solid_model.decode_blocks:
            attn_channels.append(block.resnets[0].conv1.out_channels)
        for i in range(len(attn_channels)):
            self.f2f_model.append(
                CrossAttnEncodeBlock3D(
                    in_channels=attn_channels[i],
                    out_channels=attn_channels[i],
                    temb_channels=self.face_model.time_embed_dim,
                    bbox_channels=None, #self.face_model.bbox_embed_dim,
                    num_layers=layers_per_block,
                    resnet_act_fn=act_fn,
                    attention_head_dim=attention_head_dim,
                    num_groups=norm_num_groups,
                    eps=norm_eps,
                    downsample_scale=0,
                    attn_processor_name="AttnProcessor3D_2_0_per_pixel",
                    zero_init=cross_zero_init,
                )
            )
            self.s2f_model.append(
                CrossEncodeBlock3D(
                    channels=attn_channels[i],
                    cross_channels=attn_channels[i],
                    temb_channels=self.solid_model.time_embed_dim,
                    bbox_channels=None, #self.solid_model.bbox_embed_dim,
                    num_layers=layers_per_block,
                    resnet_act_fn=act_fn,
                    num_groups=norm_num_groups,
                    eps=norm_eps,
                    kernel_size=1, stride=1, padding=0,
                    zero_init=cross_zero_init,
                )
            )
            self.f2s_model.append(
                CrossAttnEncodeBlock3D(
                    in_channels=attn_channels[i],
                    out_channels=attn_channels[i],
                    temb_channels=self.face_model.time_embed_dim,
                    bbox_channels=None, #self.face_model.bbox_embed_dim,
                    num_layers=layers_per_block,
                    resnet_act_fn=act_fn,
                    attention_head_dim=attention_head_dim,
                    num_groups=norm_num_groups,
                    eps=norm_eps,
                    downsample_scale=0,
                    attn_processor_name="AttnProcessor3D_2_0_per_pixel",
                    zero_init=cross_zero_init,
                )
            )

    def time_encode(self, timestep, bs, device):
        timesteps = timestep

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(bs, dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype, device=device)
        t_emb = self.time_embedding(t_emb)

        return t_emb

    def cross_attn(self, face_latent, solid_latent, 
                   t_emb, f2f_model, s2f_model, f2s_model, mask):
        # face_latent: bs, m, ch, n, n, n
        # solid_latent: bs, ch, n, n, n
        # t_emb: bs, ch
        # mask: bs, m
        original_face_latent = face_latent
        m = face_latent.shape[1]
        bs = face_latent.shape[0]
        ch = face_latent.shape[2]
        solid_cross_latent_bank = solid_latent
        face_cross_latent_bank = face_latent

        # extrct cross_attn latent for face_latent
        face_cross_latent = []
        face_cross_mask = []
        for m_i in range(m):
            cross_idx = [i for i in range(m) if i != m_i]
            face_cross = face_cross_latent_bank[:,cross_idx] # bs, m-1, ch, n, n, n
            face_cross_latent.append(face_cross[:,None])

            face_cross_mask.append(mask[:, cross_idx]) # bs, m-1
        face_cross_mask = torch.stack(face_cross_mask, dim=1) # bs, m, m-1
        face_cross_mask = face_cross_mask.reshape(bs*m, m-1) # bs * m, m-1
        face_cross_latent = torch.cat(face_cross_latent, 1) # bs, m, m-1, ch, n, n, n
        face_cross_latent = face_cross_latent.reshape(bs * m, *face_cross_latent.shape[2:]) # bs * m, m-1, ch, n, n, n

        # 1. f2f
        face_t_emb = t_emb[:,None].repeat(1, face_latent.shape[1], 1) # bs, m, ch
        face_t_emb = face_t_emb.reshape(-1, *face_t_emb.shape[2:]) # bs * m, ch
        face_latent = face_latent.reshape(bs * m, ch, *face_latent.shape[3:]) # bs * m, ch, n, n, n

        if f2f_model is not None:
            f2f_out = f2f_model(
                face_latent, face_t_emb, None, 
                cross_hidden_states=face_cross_latent,
                cross_attention_kwargs={'attention_mask': face_cross_mask})[0]
            f2f_out = f2f_out.reshape(bs, m, *f2f_out.shape[1:])
        else:
            f2f_out = None
        
        # 2. s2f
        if s2f_model is not None:
            solid_cross_latent = solid_cross_latent_bank[:,None].repeat(1, m, 1, 1, 1, 1) # bs, m, ch, n, n, n
            solid_cross_latent = solid_cross_latent.reshape(bs * m, ch, *solid_cross_latent.shape[3:]) # bs * m, ch, n, n, n
            s2f_out = s2f_model(
                face_latent, face_t_emb, None,
                cross_hidden_states=solid_cross_latent)[0]
            s2f_out = s2f_out.reshape(bs, m, *s2f_out.shape[1:])
        else:
            s2f_out = None
        
        # 3. f2s
        if f2s_model is not None:
            f2s_out = f2s_model(
                solid_latent, t_emb, None,
                cross_hidden_states=original_face_latent,
                cross_attention_kwargs={'attention_mask': mask})[0]
        else:
            f2s_out = None

        return f2f_out, s2f_out, f2s_out
    
    def cross_attn_block(self, face_latent, solid_latent, 
                         t_emb, layer_idx, mask):
        f2f_attn = self.f2f_model[layer_idx]
        f2s_attn = self.f2s_model[layer_idx]
        s2f_attn = self.s2f_model[layer_idx]
        f2f_out, s2f_out, f2s_out = \
            self.cross_attn(face_latent, solid_latent, 
                            t_emb, f2f_attn, s2f_attn, f2s_attn, mask)
        if f2f_out is not None and s2f_out is not None:
            face_cross_latent = (f2f_out + s2f_out) / 2
        elif f2f_out is not None:
            face_cross_latent = f2f_out
        elif s2f_out is not None:
            face_cross_latent = s2f_out
        else:
            face_cross_latent = 0
        face_latent = face_latent + face_cross_latent
        if f2s_out is not None:
            solid_latent = solid_latent + f2s_out
        
        return face_latent, solid_latent
        
    def forward(self, solid, faces, bbox, mask, timestep):
        solid = solid[:,0]
        # solid: bs, ch, n, n, n
        # faces: bs, m, ch, n, n, n
        # bbox: bs, m, 6
        # mask: bs, m
        # timestep: bs

        bs = faces.shape[0]
        m = faces.shape[1]

        # 1. time
        t_emb = self.time_encode(timestep, bs, solid.device) # bs, dim

        # 2. bbox
        bbox_emb = self.bbox_embed(bbox) # b,m,dim
        bbox_solid_emb = (bbox_emb * mask[...,None]).sum(dim=1) / mask.sum(dim=1, keepdim=True) # b,dim

        solid_latent = self.solid_model.conv_in_model(solid) # bs, ch, n, n, n
        face_latent = self.face_model.conv_in_model(faces) # bs, m, ch, n, n, n

        solid_block_res_samples = (solid_latent, )
        face_block_res_samples = (face_latent, )

        layer_idx = 0
        # down
        for encode_block_i in range(len(self.face_model.encode_blocks)):
            # cross attn
            face_latent, solid_latent = \
                self.cross_attn_block(face_latent, solid_latent, 
                                    t_emb, layer_idx, mask)

            solid_latent, solid_res_samples = self.solid_model.block_forward(
                self.solid_model.encode_blocks[encode_block_i], solid_latent, t_emb, bbox_solid_emb)
            solid_block_res_samples += solid_res_samples
            face_latent, face_res_samples = self.face_model.block_forward(
                self.face_model.encode_blocks[encode_block_i], face_latent, t_emb, bbox_emb)
            face_block_res_samples += face_res_samples
                                                              
            layer_idx += 1
        
        # mid
        for mid_block_i in range(len(self.face_model.mid_block)):
            solid_latent, _ = self.solid_model.block_forward(
                self.solid_model.mid_block[mid_block_i], solid_latent, t_emb, bbox_solid_emb)
            face_latent, _ = self.face_model.block_forward(
                self.face_model.mid_block[mid_block_i], face_latent, t_emb, bbox_emb)

            # cross attn
            face_latent, solid_latent = \
                self.cross_attn_block(face_latent, solid_latent, 
                                    t_emb, layer_idx, mask)
            layer_idx += 1
        
        # up
        for decode_block_i in range(len(self.face_model.decode_blocks)):
            solid_decode_block = self.solid_model.decode_blocks[decode_block_i]
            solid_res_samples = solid_block_res_samples[-len(solid_decode_block.resnets):]
            solid_block_res_samples = solid_block_res_samples[:-len(solid_decode_block.resnets)]
            solid_latent = self.solid_model.block_forward(
                solid_decode_block, solid_latent, t_emb, bbox_solid_emb, res_hidden_states_tuple=solid_res_samples)

            face_decode_block = self.face_model.decode_blocks[decode_block_i]
            face_res_samples = face_block_res_samples[-len(face_decode_block.resnets):]
            face_block_res_samples = face_block_res_samples[:-len(face_decode_block.resnets)]
            face_latent = self.face_model.block_forward(
                face_decode_block, face_latent, t_emb, bbox_emb, res_hidden_states_tuple=face_res_samples)

            # cross attn
            face_latent, solid_latent = \
                self.cross_attn_block(face_latent, solid_latent, 
                                    t_emb, layer_idx, mask)
            layer_idx += 1
        
        # post-process
        face_latent = self.face_model.conv_out_model(face_latent)
        solid_latent = self.solid_model.conv_out_model(solid_latent)

        solid = solid_latent + solid
        faces = faces + face_latent

        solid = solid[:,None]

        return solid, faces

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)


class UNet3DModel(nn.Module):
    def __init__(
        self,
        in_channels = 8,
        out_channels = 8,
        freq_shift = 0,
        flip_sin_to_cos = True,
        unet_param = {},
        block_channels = (224, 448, 672),
        layers_per_block = 2,
        act_fn: str = "silu",
        attention_head_dim = 8,
        norm_num_groups = 32,
        norm_eps: float = 1e-5,
        time_embed_dim = 256,
        bbox_embed_dim = 256,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.bbox_embed_dim = bbox_embed_dim

        # input
        self.conv_in = nn.Conv3d(in_channels, block_channels[0], kernel_size=3, padding=1)

        self.encode_blocks = nn.ModuleList([])
        self.mid_block = nn.ModuleList([])
        self.decode_blocks = nn.ModuleList([])

        # encode
        output_channel = block_channels[0]
        channel_start = 0
        cross_dims = [block_channels[0]]
        for i, encode_block_type in enumerate(unet_param['encode_params']['types']):
            input_channel = output_channel
            output_channel = block_channels[i+channel_start]

            if encode_block_type == "EncodeDownBlock3D":
                encode_block = EncodeDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=self.time_embed_dim,
                    bbox_channels = self.bbox_embed_dim,
                    num_layers=layers_per_block,
                    resnet_act_fn=act_fn,
                    num_groups=norm_num_groups,
                    eps=norm_eps,
                    has_attention=unet_param['encode_params']['attns'][i],
                )
                for i in range(layers_per_block):
                    cross_dims.append(output_channel)
            elif encode_block_type == "EncodeBlock3D":
                encode_block = EncodeBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=self.time_embed_dim,
                    bbox_channels = self.bbox_embed_dim,
                    num_layers=layers_per_block,
                    resnet_act_fn=act_fn,
                    attention_head_dim=attention_head_dim,
                    num_groups=norm_num_groups,
                    eps=norm_eps,
                    has_attention=unet_param['encode_params']['attns'][i],
                )
                for i in range(layers_per_block):
                    cross_dims.append(output_channel)
            else:
                raise ValueError(f"Unsupported encode block type: {encode_block_type}")

            self.encode_blocks.append(encode_block)
        
        # mid
        channel_start += len(unet_param['encode_params']['types'])
        for i, mid_block_type in enumerate(unet_param['mid_params']['types']):
            input_channel = output_channel
            output_channel = block_channels[i+channel_start]

            if mid_block_type == "EncodeBlock3D":
                mid_block = EncodeBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=self.time_embed_dim,
                    bbox_channels = self.bbox_embed_dim,
                    num_layers=layers_per_block,
                    resnet_act_fn=act_fn,
                    attention_head_dim=attention_head_dim,
                    num_groups=norm_num_groups,
                    eps=norm_eps,
                    has_attention=unet_param['mid_params']['attns'][i]
                )
            else:
                raise ValueError(f"Unsupported mid block type: {mid_block_type}")
            
            self.mid_block.append(mid_block)
        
        # decode
        channel_start += len(unet_param['mid_params']['types'])
        reverse_encode_channels = list(reversed(cross_dims))
        for i, decode_block_type in enumerate(unet_param['decode_params']['types']):
            input_channel = output_channel
            output_channel = block_channels[i+channel_start]

            num_layers = layers_per_block
            if decode_block_type == "DecodeUpBlock3D":
                decode_block = DecodeUpBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_out_channels=reverse_encode_channels[:num_layers],
                    temb_channels=self.time_embed_dim,
                    bbox_channels = self.bbox_embed_dim,
                    num_layers=num_layers,
                    resnet_act_fn=act_fn,
                    num_groups=norm_num_groups,
                    eps=norm_eps,
                    has_attention=unet_param['decode_params']['attns'][i]
                )
                reverse_encode_channels = reverse_encode_channels[num_layers:]
            elif decode_block_type == "DecodeBlock3D":
                decode_block = DecodeBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_out_channels=reverse_encode_channels[:num_layers],
                    temb_channels=self.time_embed_dim,
                    bbox_channels = self.bbox_embed_dim,
                    num_layers=num_layers,
                    resnet_act_fn=act_fn,
                    attention_head_dim=attention_head_dim,
                    num_groups=norm_num_groups,
                    eps=norm_eps,
                    has_attention=unet_param['decode_params']['attns'][i]
                )
                reverse_encode_channels = reverse_encode_channels[num_layers:]
            else:
                raise ValueError(f"Unsupported decode block type: {decode_block_type}")

            self.decode_blocks.append(decode_block)


        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_channels[-1] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_channels[-1], num_groups=num_groups_out, eps=norm_eps)
        #self.conv_norm_out = nn.Identity()
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv3d(block_channels[-1], out_channels, kernel_size=3, padding=1)


    def conv_in_model(self, sample):
        if sample.dim() == 6:
            # multiple faces in batch: bs, m, ch, n, n, n
            bs = sample.shape[0]
            m = sample.shape[1]
            sample = sample.reshape(bs * m, *sample.shape[2:])
            sample = self.conv_in(sample)
            sample = sample.reshape(bs, m, *sample.shape[1:])
        else:
            sample = self.conv_in(sample)
        return sample
    
    def block_forward(self, model, sample, t_emb, bbox_emb=None, **kwargs):
        if sample.dim() == 6:
            # multiple faces in batch: bs, m, ch, n, n, n
            bs = sample.shape[0]
            m = sample.shape[1]
            sample = sample.reshape(bs * m, *sample.shape[2:])
            # expand t_emb to match the batch size
            t_emb = t_emb[:,None].repeat(1, m, 1)
            t_emb = t_emb.reshape(bs * m, *t_emb.shape[2:])
            bbox_emb = bbox_emb.reshape(bs * m, *bbox_emb.shape[2:])
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    new_kwargs[k] = v.reshape(bs * m, *v.shape[2:])
                elif isinstance(v, tuple):
                    new_ = []
                    for v_i in v:
                        new_.append(v_i.reshape(bs * m, *v_i.shape[2:]))
                    new_kwargs[k] = tuple(new_)
                else:
                    raise ValueError(f"Unsupported type: {type(v)}")

            outs = model(sample, t_emb, bbox_emb, **new_kwargs)
            if isinstance(outs, tuple):
                sample, res_samples = outs
                sample = sample.reshape(bs, m, *sample.shape[1:])
                new_res_samples = []
                for res_sample in res_samples:
                    new_res_samples.append(res_sample.reshape(bs, m, *res_sample.shape[1:]))
                new_res_samples = tuple(new_res_samples)
                return sample, new_res_samples
            else:
                sample = outs
                sample = sample.reshape(bs, m, *sample.shape[1:])
                return sample
        
        return model(sample, t_emb, bbox_emb, **kwargs)

    def conv_out_model(self, sample):
        if sample.dim() == 6:
            # multiple faces in batch: bs, m, ch, n, n, n
            bs = sample.shape[0]
            m = sample.shape[1]
            sample = sample.reshape(bs * m, *sample.shape[2:])
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)
            sample = sample.reshape(bs, m, *sample.shape[1:])
        else:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)
        return sample

    def forward(self, sample, t_emb, bbox_emb):

        # 1. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        down_block_res_samples = (sample, )
        # 2. down
        for encode_block_i in range(len(self.encode_blocks)):
            sample, res_samples = self.block_forward(self.encode_blocks[encode_block_i], sample, t_emb, bbox_emb)
            down_block_res_samples += res_samples
        
        # 3. mid
        for mid_block_i in range(len(self.mid_block)):
            sample, _ = self.block_forward(self.mid_block[mid_block_i], sample, t_emb, bbox_emb)
        
        # 4. up
        for decode_block_i, decode_block in enumerate(self.decode_blocks):
            res_samples = down_block_res_samples[-len(decode_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(decode_block.resnets)]

            sample = self.block_forward(decode_block, sample, t_emb, bbox_emb, res_hidden_states_tuple=res_samples)

        # 5. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        return sample

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

