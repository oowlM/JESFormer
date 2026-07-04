from os import path as osp

from utils import scandir


def paired_paths_from_folder(folders, keys, filename_tmpl):
    input_folder, gt_folder = folders
    input_key, gt_key = keys
    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    if len(input_paths) != len(gt_paths):
        raise ValueError(f'{input_key} and {gt_key} datasets have different number of images.')
    paths = []
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        paths.append({f'{input_key}_path': input_path, f'{gt_key}_path': osp.join(gt_folder, gt_path)})
    return paths


def dual_paired_paths_from_folder(folders, keys, filename_tmpl):
    input_folder, gt1_folder, gt2_folder = folders
    input_key, gt1_key, gt2_key = keys
    input_paths = list(scandir(input_folder))
    gt1_paths = list(scandir(gt1_folder))
    gt2_paths = list(scandir(gt2_folder))
    if not (len(input_paths) == len(gt1_paths) == len(gt2_paths)):
        raise ValueError('Datasets have different number of images.')
    paths = []
    for gt1_path, gt2_path in zip(gt1_paths, gt2_paths):
        basename, ext = osp.splitext(osp.basename(gt1_path))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        paths.append(
            {
                f'{input_key}_path': osp.join(input_folder, input_name),
                f'{gt1_key}_path': osp.join(gt1_folder, gt1_path),
                f'{gt2_key}_path': osp.join(gt2_folder, gt2_path),
            }
        )
    return paths

