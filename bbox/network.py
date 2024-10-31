import torch
import torch.nn as nn
import math

def sincos_embedding(input, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param input: a N-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim //2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) /half
    ).to(device=input.device)
    for _ in range(len(input.size())):
        freqs = freqs[None]
    args = input.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class BBoxNet(nn.Module):
    def __init__(self, config):
        super(BBoxNet, self).__init__()
        self.config = config
        self.embed_dim = config['d_model']

        layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            norm_first=True,
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        )

        self.net = nn.TransformerEncoder(
            layer, 12, nn.LayerNorm(config['d_model'])
        )

        self.p_embed = nn.Sequential(
            nn.Linear(config['input_channel'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.SiLU(),
            nn.Linear(config['d_model'], config['d_model'])
        )

        self.time_embed = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.SiLU(),
            nn.Linear(config['d_model'], config['d_model'])
        )

        self.fc_out = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.SiLU(),
            nn.Linear(config['d_model'], config['input_channel'])
        )

    def forward(self, x, timesteps):
        # x: (bs, n, 4)
        bs = x.shape[0]
        time_embed = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        p_embed = self.p_embed(x)

        tokens = p_embed + time_embed

        output = self.net(src=tokens.permute(1, 0, 2)).transpose(0, 1)
        pred = self.fc_out(output)

        return pred

