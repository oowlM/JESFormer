import argparse
import os
from glob import glob
from os import path as osp
from natsort import natsorted
from tqdm import tqdm

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs import JESFormer

import metrics
from utils import process_without_gb, load_binary_img, load_img, save_gray_img, save_img, to_uint8_image


def parse_args():
    parser = argparse.ArgumentParser(description='JESFormer inference and optional evaluation.')
    parser.add_argument('--input_dir', default='/inputs', type=str, help='Directory of input images.')
    parser.add_argument('--weights', required=True, type=str, help='Path to weights.')
    parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results.')
    parser.add_argument('--dataset', default='LLRetina', type=str, help='Dataset name for result subfolder.')
    parser.add_argument('--gpus', type=str, default='0', help='GPU devices.')

    parser.add_argument('--test_enhance', action=argparse.BooleanOptionalAction, default=True, help='Whether to evaluate image restoration metrics.')
    parser.add_argument('--target_dir', type=str, default='', help='Directory of image restoration targets.')

    parser.add_argument('--test_segment', action=argparse.BooleanOptionalAction, default=False, help='Whether to evaluate segmentation metrics.')
    parser.add_argument('--segment_target_dir', type=str, default='', help='Directory of segmentation targets.')
    parser.add_argument('--segment_threshold', type=float, default=0.5)
    
    parser.add_argument('--test_lfps', action=argparse.BooleanOptionalAction, default=False, help='Whether to evaluate LFPS for each image.')
    parser.add_argument('--lfps_mask_dir', type=str, default='', help='Directory of vessel masks used by LFPS.')
    parser.add_argument('--lfps_weights', type=str, default='weights/RETFound_mae_natureCFP.pth', help='Path to RETFound weights for LFPS.')

    return parser.parse_args()


def load_model(weights_path):
    model = JESFormer(is_train=False)
    checkpoint = torch.load(weights_path)
    try:
        model.load_state_dict(checkpoint['params'])
    except Exception:
        new_checkpoint = {}
        for key, value in checkpoint['params'].items():
            new_checkpoint['module.' + key] = value
        model.load_state_dict(new_checkpoint)
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()
    return model


def run_inference(model, image, segment_threshold, factor=8):
    input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
    height, width = input_tensor.shape[2], input_tensor.shape[3]

    if height <= 2048 and width <= 2048:
        padded_h = ((height + factor) // factor) * factor
        padded_w = ((width + factor) // factor) * factor
        padh = padded_h - height if height % factor != 0 else 0
        padw = padded_w - width if width % factor != 0 else 0
        input_tensor = F.pad(input_tensor, (0, padw, 0, padh), 'reflect')
        denoised, segmented = model(input_tensor)
        denoised = denoised[:, :, :height, :width]
        segmented = segmented[:, :, :height, :width]
    else:
        padded_h = (height + 2 * factor) // (2 * factor) * (2 * factor)
        padded_w = (width + 2 * factor) // (2 * factor) * (2 * factor)
        padh = padded_h - height if height % (2 * factor) != 0 else 0
        padw = padded_w - width if width % (2 * factor) != 0 else 0
        input_tensor = F.pad(input_tensor, (0, padw, 0, padh), 'reflect')
        input_1 = input_tensor[:, :, :, 1::2]
        input_2 = input_tensor[:, :, :, 0::2]
        denoised_1, segmented_1 = model(input_1)
        denoised_2, segmented_2 = model(input_2)
        denoised = torch.zeros_like(input_tensor)
        segmented = torch.zeros_like(input_tensor)
        denoised[:, :, :, 1::2] = denoised_1
        denoised[:, :, :, 0::2] = denoised_2
        segmented[:, :, :, 1::2] = segmented_1
        segmented[:, :, :, 0::2] = segmented_2
        denoised = denoised[:, :, :height, :width]
        segmented = segmented[:, :, :height, :width]

    denoised = torch.clamp(denoised, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    segmented = (torch.sigmoid(segmented) > segment_threshold).float()
    segmented = torch.clamp(segmented, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    
    return denoised, segmented


def target_path_check(root_dir, image_name, suffixes=None):
    if root_dir is None:
        return None
    stem, ext = osp.splitext(image_name)
    candidates = [osp.join(root_dir, image_name)]
    if suffixes is not None:
        candidates = [osp.join(root_dir, stem + suffix) for suffix in suffixes]
    else:
        for candidate_ext in [ext, '.png', '.jpg', '.bmp', '.tif']:
            candidates.append(osp.join(root_dir, stem + candidate_ext))

    for candidate in candidates:
        if candidate is not None and osp.exists(candidate):
            return candidate
    return None


def evaluate_image_metrics(pred_rgb, target_path, lfps_mask_path=None, lfps_weights='weights/RETFound_mae_natureCFP.pth'):
    target = cv2.imread(target_path)
    process_args = [cv2.cvtColor((pred_rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)]
    if lfps_mask_path is not None:
        process_args.append((load_binary_img(lfps_mask_path) * 255).astype(np.uint8))

    masked_target, masked_preds, _ = process_without_gb(target, *process_args)
    pred = masked_preds[0].astype(np.float32) / 255.0
    target = masked_target.astype(np.float32) / 255.0
    metrics_result = {
        'psnr': metrics.calculate_psnr(pred, target, crop_border=0, input_order='HWC', test_y_channel=False),
        'ssim': metrics.calculate_ssim(pred, target, crop_border=0, input_order='HWC', test_y_channel=False),
        'lpips': metrics.calculate_lpips(pred, target, func_net='vgg'),
    }
    if lfps_mask_path is not None:
        metrics_result['lfps'] = metrics.calculate_lfps(
            masked_preds[0],
            masked_target,
            masked_preds[1],
            resume=lfps_weights,
            input_order='HWC',
            bgr_input=True,
        )
    return metrics_result


def evaluate_segment_metrics(pred_segment, target_path):
    pred_uint8 = to_uint8_image(pred_segment)
    if pred_uint8.ndim == 3 and pred_uint8.shape[2] == 1:
        pred_uint8 = np.squeeze(pred_uint8, axis=2)
    target = load_binary_img(target_path) * 255
    if target.ndim == 3 and target.shape[2] == 1:
        target = np.squeeze(target, axis=2)
    return {
        'accuracy': metrics.calculate_accuracy(pred_uint8, target),
        'precision': metrics.calculate_precision(pred_uint8, target),
        'recall': metrics.calculate_recall(pred_uint8, target),
        'dice': metrics.calculate_dice(pred_uint8, target),
        'mIoU': metrics.calculate_miou(pred_uint8, target),
        'auc_roc': metrics.calculate_auc_roc(pred_uint8, target),
    }


def summarize_metrics(metric_list):
    if not metric_list:
        return None
    keys = metric_list[0].keys()
    return {key: float(np.mean([item[key] for item in metric_list])) for key in keys}


def main():
    args = parse_args()
    gpu_list = args.gpus.replace(' ', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    if args.test_enhance and not args.target_dir:
        raise ValueError('--target_dir is required when --test_enhance is enabled.')
    if args.test_segment and not args.segment_target_dir:
        raise ValueError('--segment_target_dir is required when --test_segment is enabled.')
    if args.test_lfps and not args.lfps_mask_dir:
        raise ValueError('--lfps_mask_dir is required when --test_lfps is enabled.')

    input_paths =  natsorted(
        glob(os.path.join(args.input_dir, '*.png'))
        + glob(os.path.join(args.input_dir, '*.jpg'))
        + glob(os.path.join(args.input_dir, '*.bmp'))
        + glob(os.path.join(args.input_dir, '*.tif'))
    )

    enhance_result_dir = os.path.join(args.result_dir, args.dataset, 'enhance')
    segment_result_dir = os.path.join(args.result_dir, args.dataset, 'segment')
    os.makedirs(enhance_result_dir, exist_ok=True)
    os.makedirs(segment_result_dir, exist_ok=True)


    model = load_model(args.weights)
    print('===>Testing using weights: ', args.weights)

    enhance_metric_list = []
    segment_metric_list = []

    with torch.inference_mode():
        for inp_path in tqdm(input_paths, total=len(input_paths), position=0, leave=True, desc=f'==>Testing dataset {args.dataset}...'):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = np.float32(load_img(inp_path)) / 255.0
            denoised, segmented = run_inference(model, img, args.segment_threshold)

            image_name = osp.basename(inp_path)
            segment_name = osp.splitext(image_name)[0] + '_segmented.png'
            save_img(osp.join(enhance_result_dir, image_name), to_uint8_image(denoised))
            save_gray_img(osp.join(segment_result_dir, segment_name), to_uint8_image(segmented))

            if args.test_enhance:
                target_path = target_path_check(args.target_dir, image_name)
                if target_path is None:
                    raise FileNotFoundError(f'Image target not found for {image_name} under {args.target_dir}')
                lfps_mask_path = None
                if args.test_lfps:
                    lfps_mask_path = target_path_check(
                        args.lfps_mask_dir,
                        image_name,
                        suffixes=['.png', '.jpg', '.bmp', '.tif'],
                    )
                    if lfps_mask_path is None:
                        raise FileNotFoundError(f'LFPS mask not found for {image_name} under {args.lfps_mask_dir}')
                enhance_metric_list.append(
                    evaluate_image_metrics(
                        denoised,
                        target_path,
                        lfps_mask_path=lfps_mask_path,
                        lfps_weights=args.lfps_weights,
                    )
                )

            if args.test_segment:
                segment_target_path = target_path_check(
                    args.segment_target_dir,
                    image_name,
                    suffixes=['.png', '.jpg', '.bmp', '.tif'],
                )
                if segment_target_path is None:
                    raise FileNotFoundError(f'Segment target not found for {image_name} under {args.segment_target_dir}')
                segment_metric_list.append(evaluate_segment_metrics(segmented, segment_target_path))

    enhance_summary = summarize_metrics(enhance_metric_list)
    segment_summary = summarize_metrics(segment_metric_list)

    if enhance_summary is not None:
        print('enhancement metrics:')
        for key, value in enhance_summary.items():
            print(f'{key}: {value}')
    if segment_summary is not None:
        print('segment metrics:')
        for key, value in segment_summary.items():
            print(f'{key}: {value}')


if __name__ == '__main__':
    main()
