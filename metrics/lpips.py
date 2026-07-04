import io
import warnings
from contextlib import redirect_stdout

import lpips
import numpy as np
import torch

from metrics.metric_util import reorder_image

_LPIPS_CACHE = {}
_TORCHVISION_PRETRAINED_WARNINGS = [
    "The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.",
    "Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future.*",
]


def _build_lpips_model(net='vgg', device='cuda'):
    with warnings.catch_warnings():
        for warning_message in _TORCHVISION_PRETRAINED_WARNINGS:
            warnings.filterwarnings('ignore', message=warning_message, category=UserWarning)
        with redirect_stdout(io.StringIO()):
            return lpips.LPIPS(net=net, eval_mode=True).to(device)


def get_lpips_model(net='vgg', device='cuda'):
    key = (net, device)
    if key not in _LPIPS_CACHE:
        model = _build_lpips_model(net=net, device=device)
        model.eval()
        _LPIPS_CACHE[key] = model
    return _LPIPS_CACHE[key]


def preload_lpips_models(nets, device='cuda'):
    for net in sorted(set(nets)):
        get_lpips_model(net=net, device=device)


def calculate_lpips(img1, img2, func_net):
    if img1.shape != img2.shape:
        raise ValueError(f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    if isinstance(img1, torch.Tensor):
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        if len(img1.shape) == 3:
            img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
        elif len(img1.shape) == 2:
            img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        if len(img2.shape) == 3:
            img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
        elif len(img2.shape) == 2:
            img2 = img2.detach().cpu().numpy()

    img1 = reorder_image(img1, input_order='HWC')
    img2 = reorder_image(img2, input_order='HWC')

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    if img1.max() > 1:
        img1 = img1 / 255.0
    if img2.max() > 1:
        img2 = img2 / 255.0

    img1 = img1[:, :, ::-1].copy()
    img2 = img2[:, :, ::-1].copy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t1 = t1 * 2 - 1
    t2 = t2 * 2 - 1
    loss_fn = get_lpips_model(net=func_net, device=device)
    return float(loss_fn(t1, t2).item())
