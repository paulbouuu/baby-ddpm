import math
import torch
import torch.nn as nn



def sinusoidal_embedding(t, dim):
    """ sinusoidal embedding for time steps t """
    half = dim // 2

    freqs = torch.exp(
        -torch.arange(half, dtype=torch.float32) * (math.log(10000.0) / (half - 1))
    )

    # outer product (B,1) * (1,half) -> (B, half)
    angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)

    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return emb  # (B, dim)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, t):
        emb = sinusoidal_embedding(t, self.dim)
        return self.linear(emb)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        var = x.var(dim=1, keepdim=True, unbiased=False)
        mean = x.mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]
    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.activation = nn.SiLU()

        self.norm1 = LayerNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch)

        self.norm2 = LayerNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        # first conv
        h = self.conv1(self.activation(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.dropout(h)

        # second conv
        h = self.conv2(self.activation(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        q = self.q(h).view(B, C, H * W).transpose(1, 2) # (B, HW, C)
        k = self.k(h).view(B, C, H * W) # (B, C, HW)
        v = self.v(h).view(B, C, H * W).transpose(1, 2) # (B, HW, C)

        attn = (q @ k) * (C ** -0.5) # (B, HW, HW)
        attn = attn.softmax(dim=-1)

        attn = self.attn_dropout(attn)

        out = attn @ v # (B, HW, C)
        out = out.transpose(1, 2).view(B, C, H, W)

        out = self.proj(out)
        out = self.proj_dropout(out)

        return x + out

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, time_dim=128):
        super().__init__()

        self.time_dim = time_dim
        self.time_mlp = TimeEmbedding(time_dim)

        # downsample
        self.down1 = ResBlock(in_ch, base, time_dim)
        self.downsample1 = nn.Conv2d(base, base, 4, 2, 1)

        self.down2 = ResBlock(base, base*2, time_dim)
        self.downsample2 = nn.Conv2d(base*2, base*2, 4, 2, 1)

        self.down3 = ResBlock(base*2, base*4, time_dim)
        self.downsample3 = nn.Conv2d(base*4, base*4, 4, 2, 1)

        # bottleneck
        self.bot1 = ResBlock(base*4, base*4, time_dim)
        self.attn = SelfAttention(base*4)
        self.bot2 = ResBlock(base*4, base*4, time_dim)

        # upsample
        self.up1 = ResBlock(base*4 + base*4, base*4, time_dim)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base*4, base*4, 3, 1, 1),
        )

        self.up2 = ResBlock(base*4 + base*2, base*2, time_dim)
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base*4, base*4, 3, 1, 1),
        )

        self.up3 = ResBlock(base*2 + base, base, time_dim)
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base*2, base*2, 3, 1, 1),
        )

        self.final = nn.Conv2d(base, out_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # down
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.downsample1(d1), t_emb)
        d3 = self.down3(self.downsample2(d2), t_emb)

        # bottleneck
        b = self.bot1(self.downsample3(d3), t_emb)
        b = self.attn(b)
        b = self.bot2(b, t_emb)

        # up
        u1 = self.up1(torch.cat([self.upsample1(b), d3], dim=1), t_emb)
        u2 = self.up2(torch.cat([self.upsample2(u1), d2], dim=1), t_emb)
        u3 = self.up3(torch.cat([self.upsample3(u2), d1], dim=1), t_emb)

        return self.final(u3)