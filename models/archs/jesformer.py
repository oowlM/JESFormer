import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import attenBlock as modules
AB_DICT = {name: getattr(modules, name) for name in modules.__all__}


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_.', stacklevel=2)
    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LRelu(nn.Module):
    def forward(self, x):
        return F.leaky_relu(x, negative_slope=0.1, inplace=True)


class ResBlockWithAttn(nn.Module):
    def __init__(self, dim, dim_out, attn_heads, attn):
        super().__init__()
        self.conv_block = self._build_conv_layer(dim, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.attn = attn(dim=dim_out, num_blocks=1, dim_head=dim_out // attn_heads, heads=attn_heads)

    def _build_conv_layer(self, in_dim, out_dim, norm_cfg=True):
        layers = [nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim), nn.Conv2d(in_dim, out_dim, kernel_size=1)]
        if norm_cfg:
            layers.append(nn.GroupNorm(out_dim, out_dim))
        layers.append(LRelu())
        return nn.Sequential(*layers)

    def forward(self, x):
        hidden = self.conv_block(x)
        return self.attn(hidden + hidden + self.res_conv(x))


class HiTAS(nn.Module):
    def __init__(self, prompt_dim, dim, num_layers=3):
        super().__init__()
        self.layer_emb = nn.Embedding(num_layers, prompt_dim)
        self.mlp_prompter = nn.Sequential(nn.Linear(prompt_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)

    def forward(self, fea, task_emb, layer_id):
        layer_emb = self.layer_emb(torch.tensor(layer_id, device=fea.device))
        prompt = self.mlp_prompter(task_emb + layer_emb).unsqueeze(0)
        batch, channels, height, width = fea.shape
        fea_flat = fea.flatten(2).permute(0, 2, 1)
        query = self.wq(fea_flat)
        key = self.wk(prompt).unsqueeze(0)
        value = self.wv(prompt).unsqueeze(0)
        attn = torch.softmax((query @ key.transpose(1, 2)) * (channels ** -0.5), dim=1)
        return (attn @ value).permute(0, 2, 1).view(batch, channels, height, width)


class JESFormer(nn.Module):
    def __init__(self, is_train=True, in_channels=3, denoise_channels=3, segment_channels=1, n_feat=40, level=3, num_blocks=[1, 2, 2, 2]):
        super().__init__()
        if len(num_blocks) != level + 1:
            raise ValueError("The net depth doesn't match the num_block settings.")
        self.is_train = is_train
        self.dim = n_feat
        self.level = level
        self.dim_level = [n_feat * (2 ** i) for i in range(level + 1)]
        self.encoder_ab = AB_DICT['CW_AB']
        self.den_decoder_ab = AB_DICT['SMCW_AB']
        self.seg_decoder_ab = AB_DICT['FocusedRelu_AB']
        self.task_embedding = nn.Embedding(2, n_feat)
        self.embedding = nn.Conv2d(in_channels, self.dim, 3, 1, 1, bias=False)

        self.encoder = nn.ModuleList()
        for index in range(level):
            current_dim = self.dim_level[index]
            self.encoder.append(
                nn.ModuleList(
                    [
                        self.encoder_ab(dim=current_dim, num_blocks=num_blocks[index], dim_head=n_feat, heads=current_dim // n_feat),
                        nn.Conv2d(current_dim, current_dim * 2, 4, 2, 1, bias=False),
                        HiTAS(prompt_dim=self.dim, dim=current_dim),
                    ]
                )
            )

        self.bottleneck = self.encoder_ab(
            dim=self.dim_level[-1],
            dim_head=n_feat,
            heads=self.dim_level[-1] // n_feat,
            num_blocks=num_blocks[-1],
        )

        self.segment_decoder = nn.ModuleList()
        for index in range(level):
            current_dim = self.dim_level[level - 1 - index]
            self.segment_decoder.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(current_dim * 2, current_dim, stride=2, kernel_size=2),
                        ResBlockWithAttn(current_dim * 2, current_dim, current_dim // n_feat, attn=self.seg_decoder_ab),
                    ]
                )
            )
        self.segment_mapping = nn.Conv2d(self.dim, segment_channels, kernel_size=1)

        self.denoise_decoder = nn.ModuleList()
        for index in range(level):
            current_dim = self.dim_level[level - 1 - index]
            self.denoise_decoder.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(current_dim * 2, current_dim, stride=2, kernel_size=2),
                        nn.Conv2d(current_dim * 2, current_dim, 1, 1, bias=False),
                        self.den_decoder_ab(dim=current_dim, num_blocks=num_blocks[level - 1 - index], dim_head=n_feat, heads=current_dim // n_feat),
                    ]
                )
            )
        self.denoise_mapping = nn.Conv2d(self.dim, denoise_channels, 3, 1, 1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        fea = self.embedding(x)
        segment_task = self.task_embedding(torch.tensor(0, device=x.device))
        denoise_task = self.task_embedding(torch.tensor(1, device=x.device))

        fea_encoder, resfea_encoder = [], []
        for layer_id, (ab, downsample, prompter) in enumerate(self.encoder):
            fea_h = ab(fea)
            fea_encoder.append(prompter(fea_h, denoise_task, layer_id))
            resfea_encoder.append(prompter(fea_h, segment_task, layer_id))
            fea = downsample(fea_h)

        bottom_fea = self.bottleneck(fea)

        segfea_decoder = []
        segment_fea = bottom_fea
        for index, (upsample, block) in enumerate(self.segment_decoder):
            segment_fea = upsample(segment_fea)
            segment_fea = block(torch.cat((segment_fea, resfea_encoder[self.level - 1 - index]), dim=1))
            segfea_decoder.append(segment_fea)
        segment_out = self.segment_mapping(segment_fea)

        denoise_fea = bottom_fea
        for index, (upsample, skip, ab) in enumerate(self.denoise_decoder):
            denoise_fea = upsample(denoise_fea)
            denoise_fea = skip(torch.cat((denoise_fea, fea_encoder[self.level - 1 - index]), dim=1))
            denoise_fea = ab(denoise_fea, segfea_decoder[index])
        noise = self.denoise_mapping(denoise_fea)
        return noise + x, segment_out
