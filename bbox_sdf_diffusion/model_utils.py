# Copied or modified from diffusers
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from diffusers.models.attention_processor import Attention


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}

def get_activation(act_fn: str) -> nn.Module:
    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def get_parameter_dtype(parameter: torch.nn.Module) -> torch.dtype:
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()
        linear_cls = nn.Linear

        self.linear_1 = linear_cls(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = linear_cls(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb

def Normalize(in_channels, num_groups=32, eps=1e-5):
    #return torch.nn.InstanceNorm3d(num_features=in_channels)
    #return torch.nn.LayerNorm()
    #return torch.nn.Identity()
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps, affine=True)

class ResNetBlockEmbed(nn.Module):
    def __init__(self, in_channels, out_channels=None, 
                 temb_channels=512, bbox_channels=512,
                 act_fn="silu", num_groups=32, eps=1e-5, 
                 kernel_size=3, stride=1, padding=1, res_connection=True):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, num_groups, eps)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding)
        self.norm2 = Normalize(out_channels, num_groups, eps)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding)
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None
        
        if bbox_channels is not None:
            self.bbox_proj = nn.Linear(bbox_channels, out_channels)
        else:
            self.bbox_proj = None

        if self.in_channels != self.out_channels:
            self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                 out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding)
        
        self.nonlinearity = get_activation(act_fn)

        self.res_connection = res_connection

    def forward(self, x, temb, bbox_embed):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        if self.time_emb_proj is not None:
            temb = self.time_emb_proj(temb)
            temb = temb[:, :, None, None, None]
            h = h + temb
        if self.bbox_proj is not None:
            bbox = self.bbox_proj(bbox_embed)
            bbox = bbox[:, :, None, None, None]
            h = h + bbox

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        if self.res_connection:
            h = h + x
        else:
            h = h + x * 0
        return h

class AttnProcessor3D_2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states

        args = (scale,)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        elif input_ndim == 5:
            batch_size, channel, height, width, depth = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width * depth).transpose(1, 2)

        if encoder_hidden_states is not None:
            if encoder_hidden_states.ndim == 6:
                # cross attn
                ## bs, m, ch, n, n, n
                m = encoder_hidden_states.shape[1]
                ch = encoder_hidden_states.shape[2]
                c_height, c_width, c_depth = encoder_hidden_states.shape[3:]
                encoder_hidden_states = encoder_hidden_states.reshape(batch_size, m, ch, c_height * c_width * c_depth)
                encoder_hidden_states = encoder_hidden_states.permute(0, 1, 3, 2)
                encoder_hidden_states = encoder_hidden_states.reshape(batch_size, m * c_height * c_width * c_depth, ch)
                # bs, m*n*n*n, ch
            elif encoder_hidden_states.ndim == 5:
                # bs, ch, n, n, n
                ch = encoder_hidden_states.shape[1]
                c_height, c_width, c_depth = encoder_hidden_states.shape[2:]
                encoder_hidden_states = encoder_hidden_states.view(batch_size, ch, c_height * c_width * c_depth)
                encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1) # bs, n*n*n, ch
        else:
            batch_size, sequence_length, _ = hidden_states.shape 

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        elif input_ndim == 5:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width, depth)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class AttnProcessor3D_2_0_per_pixel:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states

        args = (scale,)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            raise NotImplementedError("Per-pixel attention is not implemented for 3D")
        elif input_ndim == 5:
            batch_size, channel, height, width, depth = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width * depth).transpose(1, 2)
            # bs, nxnxn, ch
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            # bs*nxnxn, ch
            hidden_states = hidden_states.unsqueeze(1) # bs*nxnxn, 1, ch

        if encoder_hidden_states is not None:
            if encoder_hidden_states.ndim == 6:
                # cross attn
                ## bs, m, ch, n, n, n
                m = encoder_hidden_states.shape[1]
                ch = encoder_hidden_states.shape[2]
                c_height, c_width, c_depth = encoder_hidden_states.shape[3:]
                encoder_hidden_states = encoder_hidden_states.reshape(batch_size, m, ch, c_height * c_width * c_depth)
                encoder_hidden_states = encoder_hidden_states.permute(0, 3, 1, 2) # bs, n*n*n, m, ch
                encoder_hidden_states = encoder_hidden_states.reshape(-1, m, ch)
                # bs*n*n*n, m, ch
            elif encoder_hidden_states.ndim == 5:
                raise NotImplementedError("Per-pixel attention is not implemented for 3D")
                # bs, ch, n, n, n
                ch = encoder_hidden_states.shape[1]
                c_height, c_width, c_depth = encoder_hidden_states.shape[2:]
                encoder_hidden_states = encoder_hidden_states.view(batch_size, ch, c_height * c_width * c_depth)
                encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1) # bs, n*n*n, ch
        else:
            batch_size, sequence_length, _ = hidden_states.shape 

        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = attention_mask[:,None].repeat(1, c_height*c_width*c_depth, 1)
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) # BS, 1, source_length, target_length
            #attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            #attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
            pass

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # check to_q pass params whether need args
        try:
            query = attn.to_q(hidden_states, *args)
        except:
            query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        try:
            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)
        except:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(query.shape[0], -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(key.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(value.shape[0], -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if True:
            # use small batch size to avoid runtime error (larger than block thread num)
            hidden_states_list = []
            for i in range(0, query.shape[0], 65000):
                query_ = query[i:i+65000]
                key_ = key[i:i+65000]
                value_ = value[i:i+65000]
                mask_ = attention_mask[i:i+65000] if attention_mask is not None else None
                hidden_states_ = F.scaled_dot_product_attention(
                    query_, key_, value_, attn_mask=mask_, dropout_p=0.0, is_causal=False
                )
                hidden_states_list.append(hidden_states_)
            hidden_states = torch.cat(hidden_states_list, dim=0)

        #hidden_states = F.scaled_dot_product_attention(
        #    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        #)

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        # bs*nxnxn, 1, ch

        # linear proj
        try:
            hidden_states = attn.to_out[0](hidden_states, *args)
        except:
            hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states) # bs*nxnxn, 1, ch

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        elif input_ndim == 5:
            hidden_states = hidden_states[:,0].reshape(batch_size, -1, hidden_states.shape[-1])
            # bs, nxnxn, ch
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width, depth)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class EncodeBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int, out_channels: int, temb_channels: int, bbox_channels: int,
        num_layers: int = 1, resnet_act_fn: str = "swish", attention_head_dim: int = 1,
        num_groups: int = 32, eps: float = 1e-5, zero_init: bool = False,
        attention_residual: bool = True, has_attention: bool = True,
        use_position_encoding: bool = False, kernel_size: int = 3,
        stride: int = 1, padding: int = 1, 
        attn_processor_name = "AttnProcessor3D_2_0", # AttnProcessor3D_2_0_per_pixel
        res_connection: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        if has_attention:
            attn_processor = globals()[attn_processor_name]

        self.use_position_encoding = use_position_encoding
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResNetBlockEmbed(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    bbox_channels=bbox_channels,
                    act_fn=resnet_act_fn,
                    num_groups=num_groups,
                    eps=eps,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    res_connection=res_connection
                )
            )
            if has_attention:
                attentions.append(
                    Attention(
                        out_channels,
                        heads=out_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        residual_connection=attention_residual,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                        processor=attn_processor()
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if zero_init:
            for resnet in self.resnets:
                resnet.conv2.weight.data.zero_()
                resnet.conv2.bias.data.zero_()
            for attn in self.attentions:
                attn.to_out[0].weight.data.zero_()
                attn.to_out[0].bias.data.zero_()
                
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb = None,
        bbox_embed = None,
        encoder_hidden_states = None,
        cross_attention_kwargs = None,
    ):
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        lora_scale = cross_attention_kwargs.get("scale", 1.0)

        output_states = ()

        for model_i in range(len(self.resnets)):
            resnet = self.resnets[model_i]
            cross_attention_kwargs.update({"scale": lora_scale})
            hidden_states = resnet(hidden_states, temb, bbox_embed)
            if len(self.attentions) > 0:
                attn = self.attentions[model_i]
                if self.use_position_encoding:
                    encoder_hidden_states = None
                hidden_states = attn(hidden_states, encoder_hidden_states, **cross_attention_kwargs)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

class EncodeDownBlock3D(EncodeBlock3D):
    def __init__(self, 
                 in_channels: int, out_channels: int, 
                 temb_channels: int, bbox_channels: int,
                 num_layers: int = 1, 
                 resnet_act_fn: str = "swish", attention_head_dim: int = 1, 
                 num_groups: int = 32, eps: float = 0.00001, 
                 zero_init: bool = False, attention_residual: bool = True, 
                 has_attention: bool = True):
        super().__init__(in_channels, out_channels, temb_channels, bbox_channels, num_layers, resnet_act_fn, attention_head_dim, num_groups, eps, zero_init, attention_residual, has_attention)
        self.downsamplers = Downsample(out_channels, out_channels)
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb = None,
        bbox_embed = None,
        encoder_hidden_states = None,
        cross_attention_kwargs = None,
    ):
        hidden_states, output_states = super().forward(hidden_states, temb, bbox_embed, encoder_hidden_states, cross_attention_kwargs)
        hidden_states = self.downsamplers(hidden_states)
        #output_states = output_states + (hidden_states,)
        return hidden_states, output_states

class DecodeBlock3D(nn.Module):
    def __init__(self, 
                 in_channels: int, out_channels: int,
                 prev_out_channels: int, temb_channels: int,
                 bbox_channels: int,
                 num_layers: int = 1, resnet_act_fn: str = "swish",
                 attention_head_dim: int = 1,
                 num_groups: int = 32,
                 eps: float = 1e-5,
                 zero_init: bool = False,
                 attention_residual: bool = True,
                 has_attention: bool = True,
                 kernel_size: int = 3, padding: int = 1, stride: int = 1,
                 attn_processor_name = "AttnProcessor3D_2_0", # AttnProcessor3D_2_0_per_pixel
                 ):
        super().__init__()
        resnets = []
        attentions = []

        if has_attention:
            attn_processor = globals()[attn_processor_name]

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.prev_out_channels = prev_out_channels
        for i in range(num_layers):
            #res_skip_channels = in_channels if i == 0 else out_channels
            res_skip_channels = prev_out_channels[i]
            resnet_in_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResNetBlockEmbed(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    bbox_channels=bbox_channels,
                    act_fn=resnet_act_fn,
                    num_groups=num_groups,
                    eps=eps,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            if has_attention:
                attentions.append(
                    Attention(
                        out_channels,
                        heads=out_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        residual_connection=attention_residual,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                        processor=attn_processor()
                    )
                )
        
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if zero_init:
            for attn in self.attentions:
                attn.to_out[0].weight.data.zero_()
                attn.to_out[0].bias.data.zero_()
    
    def forward(self, 
                hidden_states: torch.FloatTensor, 
                temb = None, 
                bbox_embed = None,
                res_hidden_states_tuple = None, 
                encoder_hidden_states = None, 
                cross_attention_kwargs = None,):
        
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        lora_scale = cross_attention_kwargs.get("scale", 1.0)
        
        for model_i in range(len(self.resnets)):
            resnet = self.resnets[model_i]
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb, bbox_embed)
            if len(self.attentions) > 0:
                attn = self.attentions[model_i]
                hidden_states = attn(hidden_states, encoder_hidden_states, **cross_attention_kwargs)
        
        return hidden_states

class DecodeUpBlock3D(DecodeBlock3D):
    def __init__(self, 
                 in_channels: int, out_channels: int,
                 prev_out_channels: int, temb_channels: int,
                 bbox_channels: int,
                 num_layers: int = 1, resnet_act_fn: str = "swish",
                 attention_head_dim: int = 1,
                 num_groups: int = 32,
                 eps: float = 1e-5,
                 zero_init: bool = False,
                 attention_residual: bool = True,
                 has_attention: bool = True):
        super().__init__(in_channels, out_channels, prev_out_channels, temb_channels, bbox_channels, num_layers, resnet_act_fn, attention_head_dim, num_groups, eps, zero_init, attention_residual, has_attention)
        self.upsamplers = Upsample(in_channels, in_channels)
    
    def forward(self, 
                hidden_states: torch.FloatTensor, 
                temb = None, 
                bbox_embed = None,
                res_hidden_states_tuple=None, 
                encoder_hidden_states = None, 
                cross_attention_kwargs = None,):
        hidden_states = self.upsamplers(hidden_states)
        hidden_states = super().forward(hidden_states, temb, bbox_embed, res_hidden_states_tuple, encoder_hidden_states, cross_attention_kwargs)
        return hidden_states

class Upsample(nn.Module):
    def __init__(self, dims, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv3d(dims, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))
    
class Downsample(nn.Module):
    def __init__(self, dims, dim):
        super().__init__()
        self.conv = nn.Conv3d(dims, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

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

class CrossAttnEncodeBlock3D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        temb_channels: int, bbox_channels: int,
        num_layers: int = 1, resnet_act_fn: str = "swish",
        attention_head_dim: int = 1, num_groups: int = 32,
        eps: float = 1e-5, zero_init: bool = False, downsample_scale: int = 2,
        kernel_size: int = 3, stride: int = 1, padding: int = 1,
        attn_processor_name = "AttnProcessor3D_2_0", # AttnProcessor3D_2_0_per_pixel
    ):
        super().__init__()
        self.cross_block = EncodeBlock3D(
            in_channels, out_channels, temb_channels, bbox_channels,
            num_layers, resnet_act_fn, attention_head_dim, 
            num_groups, eps, zero_init, 
            attention_residual=False, has_attention=True, 
            kernel_size=kernel_size, stride=stride, padding=padding,
            attn_processor_name=attn_processor_name,
            res_connection=False)
        
        self.downsampler = []
        for i in range(downsample_scale):
            self.downsampler.append(Downsample(out_channels, out_channels))
        self.downsampler = nn.Sequential(*self.downsampler)
    
    def forward(self, hidden_states, temb, bbox_embed, cross_hidden_states, cross_attention_kwargs=None):
        # hidden_states: (B, C, H, W, D)
        # cross_hidden_states: (B, M, C, H, W, D)
        bs = hidden_states.shape[0]
        m = cross_hidden_states.shape[1]
        cross_hidden_states = cross_hidden_states.reshape(bs*m, *cross_hidden_states.shape[2:])
        cross_hidden_states = self.downsampler(cross_hidden_states)
        cross_hidden_states = cross_hidden_states.reshape(bs, m, *cross_hidden_states.shape[1:])

        return self.cross_block(hidden_states, temb, bbox_embed, cross_hidden_states, cross_attention_kwargs)

class CrossEncodeBlock3D(nn.Module):
    def __init__(
        self, channels: int, cross_channels: int, temb_channels: int, bbox_channels: int,
        num_layers: int = 1, resnet_act_fn: str = "swish", attention_head_dim: int = 1,
        num_groups: int = 32, eps: float = 1e-5, zero_init: bool = False,
        kernel_size: int = 3, stride: int = 1, padding: int = 1,
    ):
        super().__init__()
        in_channels = channels + cross_channels
        self.cross_block = EncodeBlock3D(
            in_channels, channels, temb_channels, bbox_channels,
            num_layers, resnet_act_fn, attention_head_dim, 
            num_groups, eps, zero_init, kernel_size=kernel_size, 
            stride=stride, padding=padding, 
            attention_residual=False, has_attention=False,
            res_connection=False)
        
    def forward(self, hidden_states, temb, bbox_embed, cross_hidden_states):
        # hidden_states: (B, C1, H, W, D)
        # cross_hidden_states: (B, C2, H, W, D)
        input_states = torch.cat([hidden_states, cross_hidden_states], dim=1)
        output_states, out_blocks = self.cross_block(input_states, temb, bbox_embed)
        return output_states, out_blocks

