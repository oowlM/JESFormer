import random

import cv2
import numpy as np


def dual_paired_random_crop(img_gts1, img_gts2, img_lqs, lq_patch_size, scale, gt_path1, gt_path2):
    if not isinstance(img_gts1, list):
        img_gts1 = [img_gts1]
    if not isinstance(img_gts2, list):
        img_gts2 = [img_gts2]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq = img_lqs[0].shape[:2]
    h_gt1, w_gt1 = img_gts1[0].shape[:2]
    h_gt2, w_gt2 = img_gts2[0].shape[:2]
    gt_patch_size = int(lq_patch_size * scale)

    if h_gt1 != h_lq * scale or w_gt1 != w_lq * scale:
        raise ValueError(f'Scale mismatches for {gt_path1}.')
    if h_gt2 != h_lq * scale or w_gt2 != w_lq * scale:
        raise ValueError(f'Scale mismatches for {gt_path2}.')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size ({lq_patch_size}, {lq_patch_size}).')

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)
    img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts1 = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts1]
    img_gts2 = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts2]
    if len(img_gts1) == 1:
        img_gts1 = img_gts1[0]
    if len(img_gts2) == 1:
        img_gts2 = img_gts2[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts1, img_gts2, img_lqs


def dual_paired_resize(img_gts1, img_gts2, img_lqs, lq_patch_size, scale, gt1_path, gt2_path):
    if not isinstance(img_gts1, list):
        img_gts1 = [img_gts1]
    if not isinstance(img_gts2, list):
        img_gts2 = [img_gts2]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt1, w_gt1, _ = img_gts1[0].shape
    h_gt2, w_gt2, _ = img_gts2[0].shape
    gt_patch_size = int(lq_patch_size * scale)

    if h_gt1 != h_lq * scale or w_gt1 != w_lq * scale:
        raise ValueError(f'Scale mismatches for {gt1_path}.')
    if h_gt2 != h_lq * scale or w_gt2 != w_lq * scale:
        raise ValueError(f'Scale mismatches for {gt2_path}.')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size ({lq_patch_size}, {lq_patch_size}).')

    img_lqs = [np.resize(v, (gt_patch_size, gt_patch_size, v.shape[2])) for v in img_lqs]
    img_gts1 = [np.resize(v, (gt_patch_size, gt_patch_size, v.shape[2])) for v in img_gts1]
    img_gts2 = [np.resize(v, (gt_patch_size, gt_patch_size, v.shape[2])) for v in img_gts2]
    if len(img_gts1) == 1:
        img_gts1 = img_gts1[0]
    if len(img_gts2) == 1:
        img_gts2 = img_gts2[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts1, img_gts2, img_lqs


def random_augmentation(*imgs):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    def _augment(img):
        if hflip:
            cv2.flip(img, 1, img)
        if vflip:
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return tuple(_augment(img) for img in imgs)

