import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

__all__ = ['CW_AB', 'SMCW_AB', 'FocusedRelu_AB']


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm: Optional[str] = 'bn',
        act_func: Optional[str] = 'relu',
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = nn.GroupNorm(1, out_channels)
        else:
            self.norm = None

        if act_func == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_func == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        elif act_func == 'hswish':
            self.act = nn.Hardswish(inplace=True)
        elif act_func == 'silu':
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class CW_LA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 3, 1)
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_inp, k_inp, v_inp))
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c).permute(0, 3, 1, 2)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2))
        return out_c + out_p


class CW_AB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([CW_LA(dim=dim, dim_head=dim_head, heads=heads), PreNorm(dim, FeedForward(dim=dim))]))

    def forward(self, x):
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SMCW_LA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.guide_to_q = nn.Sequential(nn.Linear(dim, dim_head * heads, bias=False), nn.GELU())
        self.guide_to_k = nn.Sequential(nn.Linear(dim, dim_head * heads, bias=False), nn.GELU())
        self.guide_to_v = nn.Sequential(nn.Linear(dim, dim_head * heads, bias=False), nn.GELU())
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x_in, guide):
        x_in = x_in.permute(0, 2, 3, 1)
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        guide = guide.permute(0, 2, 3, 1).reshape(b, h * w, c)
        g_q = self.guide_to_q(guide)
        g_k = self.guide_to_k(guide)
        g_v = self.guide_to_v(guide)

        q = rearrange(q_inp * g_q, 'b n (h d) -> b h n d', h=self.num_heads).transpose(-2, -1)
        k = rearrange(k_inp * g_k, 'b n (h d) -> b h n d', h=self.num_heads).transpose(-2, -1)
        v = rearrange(v_inp * g_v, 'b n (h d) -> b h n d', h=self.num_heads).transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c).permute(0, 3, 1, 2)
        out_p = self.pos_emb(x_in.reshape(b, h, w, c).permute(0, 3, 1, 2))
        return out_c + out_p


class SMCW_AB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([SMCW_LA(dim=dim, dim_head=dim_head, heads=heads), PreNorm(dim, FeedForward(dim=dim))]))

    def forward(self, x, guide):
        for attn, ff in self.blocks:
            x = attn(x, guide) + x
            x = ff(x) + x
        return x


class FocusedRelu_LA(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        linear=True,
        focusing_factor=3,
        kernel_size=5,
    ):
        super().__init__()
        del qk_scale
        assert dim % heads == 0, f'dim {dim} should be divided by num_heads {heads}.'
        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size, groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, dim)))

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 3, 1)
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        _, n, _ = x.shape
        q = self.q(x)
        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
                x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
            else:
                kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(self.pool(x_)).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n_k = k.shape[1]

        k = k + self.positional_encoding
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** self.focusing_factor
        k = k ** self.focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(b, n_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(b, n_k, self.num_heads, -1).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n_k ** -0.5)) @ (v * (n_k ** -0.5))
        x = q @ kv * z

        if self.sr_ratio > 1 or self.linear:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(b * self.num_heads, -1, n_k), size=n, mode='linear').reshape(b, self.num_heads, -1, n).transpose(-2, -1)
        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.reshape(b * self.num_heads, h, w, -1).permute(0, 3, 1, 2)
        x = x + self.dwc(v).reshape(b, c, n).permute(0, 2, 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


class FocusedRelu_AB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([FocusedRelu_LA(dim=dim, dim_head=dim_head, heads=heads), FeedForward(dim=dim)]))

    def forward(self, x):
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        return x
