import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.losses.loss_util import (
    bce_with_logits_loss,
    create_window,
    cross_entropy_loss,
    gradient,
    l1_loss,
    map_ssim,
    mse_loss,
)
from models.losses.vgg_arch import VGGFeatureExtractor

_reduction_modes = ['none', 'mean', 'sum']


class CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight=1.0, window_size=None, reduction='mean', eps=1e-3):
        super().__init__()
        self.eps = eps
        self.loss_weight = loss_weight
        self.reduction = reduction
        if window_size is not None:
            self.pool = nn.AvgPool2d(window_size)

    def forward(self, x, y):
        if hasattr(self, 'pool'):
            x, y = self.pool(x), self.pool(y)
        diff = x - y
        if self.reduction == 'sum':
            loss = torch.sum(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        else:
            loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return self.loss_weight * loss


class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        kernel = torch.tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.register_buffer('kernel', torch.matmul(kernel.t(), kernel).unsqueeze(0).repeat(3, 1, 1, 1))
        self.weight = loss_weight
        self.reduction = reduction

    def conv_gauss(self, img):
        _, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=self.kernel.shape[0])

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        return current - filtered

    def forward(self, x, y):
        return mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y), reduction=self.reduction) * self.weight


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        loss_weight,
        layer_weights=None,
        vgg_type='vgg19',
        use_input_norm=True,
        range_norm=True,
        perceptual_weight=1.0,
        style_weight=0.0,
        criterion='l1',
    ):
        super().__init__()
        if layer_weights is None:
            layer_weights = {'conv1_2': 0.25, 'conv2_2': 0.25, 'conv3_4': 0.25, 'conv4_4': 0.25}
        self.loss_weight = loss_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

        self.criterion_type = criterion
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean')
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif criterion == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def _gram_mat(self, x):
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        return features.bmm(features_t) / (c * h * w)

    def forward(self, x, gt):
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        percep_loss = 0
        if self.perceptual_weight > 0:
            for key in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[key] - gt_features[key], p='fro') * self.layer_weights[key]
                else:
                    percep_loss += self.criterion(x_features[key], gt_features[key]) * self.layer_weights[key]
            percep_loss *= self.perceptual_weight
        style_loss = 0
        if self.style_weight > 0:
            for key in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[key]) - self._gram_mat(gt_features[key]),
                        p='fro',
                    ) * self.layer_weights[key]
                else:
                    style_loss += self.criterion(
                        self._gram_mat(x_features[key]),
                        self._gram_mat(gt_features[key]),
                    ) * self.layer_weights[key]
            style_loss *= self.style_weight
        return self.loss_weight * (percep_loss + style_loss)


class BCEFocalWithLogitsLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=0.9, gamma=0.5, reduction='mean'):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        if pred.dim() == 4 and pred.shape[1] == 1 and target.dim() == 4 and target.shape[1] != 1 and target.shape[-1] == 1:
            target = target.permute(0, 3, 1, 2).contiguous()
        if pred.shape != target.shape:
            raise ValueError(f'shape mismatch: p{pred.shape}, target{target.shape}')
        prob = torch.sigmoid(pred)
        loss = (
            -self.alpha * (1 - prob) ** self.gamma * target * torch.log(prob)
            - (1 - self.alpha) * prob ** self.gamma * (1 - target) * torch.log(1 - prob)
        )
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss * self.loss_weight

