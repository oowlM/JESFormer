import math
import os

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid


def img2tensor(imgs, bgr2rgb=True, dtype='float32'):
    def _totensor(img, bgr2rgb, dtype):
        if img.ndim == 3 and img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img)
        if dtype == 'float32':
            return tensor.float()
        if dtype == 'long':
            return tensor.long()
        raise ValueError(f'Unsupported dtype: {dtype}')

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, dtype) for img in imgs]
    return _totensor(imgs, bgr2rgb, dtype)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for item in tensor:
        item = item.squeeze(0).float().detach().cpu().clamp_(*min_max)
        item = (item - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = item.dim()
        if n_dim == 4:
            img_np = make_grid(item, nrow=int(math.sqrt(item.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = item.numpy().transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = np.squeeze(img_np, axis=2)
            elif rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = item.numpy()
        else:
            raise TypeError(f'Unsupported tensor dimension: {n_dim}')
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        result.append(img_np.astype(out_type))
    return result[0] if len(result) == 1 else result


def imfrombytes(content, flag='color', float32=False):
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED,
        'binary': cv2.IMREAD_GRAYSCALE,
    }
    img = cv2.imdecode(img_np, imread_flags[flag])
    if flag == 'binary':
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        img = np.expand_dims(img, axis=2)
    if float32:
        img = img.astype(np.float32) / 255.0
    return img


def padding(img_lq, img_gt, gt_size):
    h, w = img_lq.shape[:2]
    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt
    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    return img_lq, img_gt


def dual_padding(img_lq, img1_gt, img2_gt, gt_size):
    h, w = img_lq.shape[:2]
    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    if h_pad == 0 and w_pad == 0:
        return img_lq, img1_gt, img2_gt
    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img1_gt = cv2.copyMakeBorder(img1_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img2_gt = cv2.copyMakeBorder(img2_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img1_gt.ndim == 2:
        img1_gt = np.expand_dims(img1_gt, axis=2)
    if img2_gt.ndim == 2:
        img2_gt = np.expand_dims(img2_gt, axis=2)
    return img_lq, img1_gt, img2_gt


def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        os.makedirs(os.path.abspath(os.path.dirname(file_path)), exist_ok=True)
    return cv2.imwrite(file_path, img, params)



def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def to_uint8_image(image):
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)


def load_binary_img(filepath):
    return (cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) > 0).astype(np.int64)


def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)
