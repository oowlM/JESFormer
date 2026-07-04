import cv2
import numpy as np
import torch

from metrics.lpips import calculate_lpips
from metrics.metric_util import reorder_image, to_y_channel


def _to_numpy_image(img, input_order='HWC'):
    if isinstance(img, torch.Tensor):
        if len(img.shape) == 4:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.detach().cpu().numpy().transpose(1, 2, 0)
        elif len(img.shape) == 2:
            img = img.detach().cpu().numpy()
        else:
            raise ValueError(f'Unsupported tensor shape: {img.shape}')
    img = reorder_image(img, input_order=input_order)
    return img


def _normalize_to_unit(img):
    img = img.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    return img


def calculate_psnr(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    if img1.shape != img2.shape:
        raise ValueError(f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    img1 = _to_numpy_image(img1, input_order=input_order)
    img2 = _to_numpy_image(img2, input_order=input_order)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    img1 = _normalize_to_unit(img1)
    img2 = _normalize_to_unit(img2)
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10(1 / (mse + 1e-8))


def _ssim_single_channel(img1, img2, data_range=1.0, k1=0.01, k2=0.03, win_size=11, sigma=1.5):
    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError('_ssim_single_channel expects 2D arrays.')
    if min(img1.shape[0], img1.shape[1]) < win_size:
        raise ValueError(f'Input image is too small for win_size={win_size}. Got shape={img1.shape}.')

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    kernel = cv2.getGaussianKernel(win_size, sigma)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REFLECT)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 * img1, -1, window, borderType=cv2.BORDER_REFLECT) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, window, borderType=cv2.BORDER_REFLECT) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=cv2.BORDER_REFLECT) - mu1_mu2
    sigma1_sq = np.maximum(sigma1_sq, 0.0)
    sigma2_sq = np.maximum(sigma2_sq, 0.0)
    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def calculate_ssim(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    if img1.shape != img2.shape:
        raise ValueError(f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    img1 = _to_numpy_image(img1, input_order=input_order)
    img2 = _to_numpy_image(img2, input_order=input_order)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y_channel:
        img1 = _normalize_to_unit(to_y_channel(img1))
        img2 = _normalize_to_unit(to_y_channel(img2))
        return _ssim_single_channel(img1[..., 0], img2[..., 0], data_range=1.0)

    img1 = _normalize_to_unit(img1)
    img2 = _normalize_to_unit(img2)
    if img1.ndim == 2:
        return _ssim_single_channel(img1, img2, data_range=1.0)
    if img1.ndim == 3:
        if img1.shape[2] == 1:
            return _ssim_single_channel(np.squeeze(img1, axis=2), np.squeeze(img2, axis=2), data_range=1.0)
        if img1.shape[2] == 3:
            img1_gray = cv2.cvtColor(img1.astype(np.float32), cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2.astype(np.float32), cv2.COLOR_RGB2GRAY)
            return _ssim_single_channel(img1_gray, img2_gray, data_range=1.0)
    raise ValueError('Wrong input image dimensions. Expected (H, W), (H, W, 1), or (H, W, 3).')
