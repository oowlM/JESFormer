import functools
from math import exp

import torch
from torch.autograd import Variable
from torch.nn import functional as F


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()
    return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    if weight is not None:
        if weight.dim() != loss.dim():
            raise ValueError('Weight and loss should have the same dimensions.')
        if weight.size(1) != 1 and weight.size(1) != loss.size(1):
            raise ValueError('Weight shape is incompatible with loss shape.')
        loss = loss * weight
    if weight is None or reduction == 'sum':
        return reduce_loss(loss, reduction)
    if reduction == 'mean':
        denom = weight.sum() if weight.size(1) > 1 else weight.sum() * loss.size(1)
        return loss.sum() / denom
    return loss


def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        loss = loss_func(pred, target, **kwargs)
        return weight_reduce_loss(loss, weight, reduction)

    return wrapper


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def cross_entropy_loss(pred, target):
    return F.cross_entropy(pred, target, reduction='none')


@weighted_loss
def bce_with_logits_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target, reduction='none')


def gradient(input_tensor, direction):
    _, channels, _, _ = input_tensor.shape
    kernel_x = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32, device=input_tensor.device)
    kernel_y = kernel_x.transpose(0, 1)
    kernel_x = kernel_x.view(1, 1, 3, 3).expand(1, channels, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3).expand(1, channels, 3, 3)
    kernel = kernel_x if direction == 'x' else kernel_y
    gradient_orig = torch.abs(torch.conv2d(input_tensor, kernel))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    return torch.div(gradient_orig - grad_min, grad_max - grad_min + 1e-4)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / torch.sum(gauss)


def create_window(window_size, channel=1):
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return Variable(window_2d.expand(channel, 1, window_size, window_size).contiguous())


def map_ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

