import os
import random
import time
from os import path as osp

import numpy as np
import torch

from utils.distributed import master_only
from utils.logger import get_root_logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path, rename_flag):
    if osp.exists(path) and rename_flag:
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


@master_only
def make_exp_dirs(opt):
    path_opt = opt['path'].copy()
    opt['rename_flag'] = False
    try:
        models_dir = osp.join(opt['path']['experiments_root'], 'models')
        opt['rename_flag'] = len(os.listdir(models_dir)) != 0
    except Exception:
        logger = get_root_logger()
        logger.info('rename_flag False')

    mkdir_and_rename(path_opt.pop('experiments_root'), opt['rename_flag'])
    for key, path in path_opt.items():
        if ('strict_load' not in key) and ('pretrain_network' not in key) and ('resume' not in key):
            os.makedirs(path, exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(path, suffix, recursive):
        for entry in os.scandir(path):
            if not entry.name.startswith('.') and entry.is_file():
                return_path = entry.path if full_path else osp.relpath(entry.path, root)
                if suffix is None or return_path.endswith(suffix):
                    yield return_path
            elif recursive:
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def load_resume_state(opt):
    resume_state_path = opt['path'].get('resume_state')
    if resume_state_path is None:
        training_state_dir = opt['path']['training_states']
        if osp.isdir(training_state_dir):
            state_files = []
            for name in os.listdir(training_state_dir):
                if name.endswith('.state'):
                    stem = name[:-6]
                    if stem.isdigit():
                        state_files.append((int(stem), osp.join(training_state_dir, name)))
            if state_files:
                resume_state_path = max(state_files, key=lambda item: item[0])[1]
    opt['path']['resume_state'] = resume_state_path


    if resume_state_path is None:
        return None

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        map_location = lambda storage, loc: storage.cuda(device_id)
    else:
        map_location = 'cpu'
    return torch.load(resume_state_path, map_location=map_location)


def check_resume(opt, resume_iter):
    logger = get_root_logger()
    if opt['path']['resume_state']:
        networks = [key for key in opt.keys() if key.startswith('network_')]
        flag_pretrain = any(opt['path'].get(f'pretrain_{network}') is not None for network in networks)
        if flag_pretrain:
            logger.warning('pretrain_network path will be ignored during resuming.')
        for network in networks:
            name = f'pretrain_{network}'
            basename = network.replace('network_', '')
            ignore = opt['path'].get('ignore_resume_networks')
            if ignore is None or basename not in ignore:
                opt['path'][name] = osp.join(opt['path']['models'], f'net_{basename}_{resume_iter}.pth')
                logger.info(f"Set {name} to {opt['path'][name]}")

