from collections import OrderedDict
from os import path as osp

import yaml


def ordered_yaml():
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, is_train=True):
    with open(opt_path, mode='r') as handle:
        loader, _ = ordered_yaml()
        opt = yaml.load(handle, Loader=loader)

    opt['is_train'] = is_train
    opt['name'] = osp.basename(opt_path).split('.')[0]

    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase.split('_')[0]
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        for key in ['dataroot_denoise_gt', 'dataroot_segment_gt', 'dataroot_lq']:
            if dataset.get(key) is not None:
                dataset[key] = osp.expanduser(dataset[key])

    for key, value in opt['path'].items():
        if value is not None and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(value)

    opt['path']['root'] = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')
    return opt


def dict2str(opt, indent_level=1):
    msg = '\n'
    for key, value in opt.items():
        if isinstance(value, dict):
            msg += ' ' * (indent_level * 2) + key + ':['
            msg += dict2str(value, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + key + ': ' + str(value) + '\n'
    return msg
