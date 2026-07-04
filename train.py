import argparse
import datetime
import math
import os
import random
import time
import torch
import torch.distributed as dist
from os import path as osp
import numpy as np
from data import CPUPrefetcher, CUDAPrefetcher, EnlargedSampler, create_dataloader, create_dataset
from metrics import preload_lpips_models
from models import create_model
from utils import (MessageLogger, load_resume_state, check_resume, get_dist_info, parse,
                   init_loggers, init_dist, make_exp_dirs, mkdir_and_rename, set_random_seed)


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt', type=str, default='options/JESFormer_LLRetina.yaml', help='Path to option YAML file.')
    parser.add_argument('--gpu_id', type=str, default="0", help='GPU devices, e.g. "0" or "0,1"')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # GPU selection
    # - Single-process (launcher=none): allow setting CUDA_VISIBLE_DEVICES via --gpu_id
    # - DDP (launcher!=none): DO NOT override CUDA_VISIBLE_DEVICES here; torchrun
    #   will manage device mapping via LOCAL_RANK.
    if args.launcher == 'none':
        gpu_list = args.gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    # distributed settings
    else:
        opt['dist'] = True
        init_dist(args.launcher)
        print(f'==> Initializing {args.launcher} distributed training ... ')

        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        torch.cuda.set_device(local_rank)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)

            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                                   opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.floor(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        elif phase == 'test':
            pass
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True

    resume_state = load_resume_state(opt)

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if (opt['logger'].get('use_tb_logger')) and ('debug' not in opt['name']) and (opt['rank'] == 0):
            mkdir_and_rename(
                osp.join(opt['path']['experiments_root'], 'tb_logger'), opt['rename_flag'])

    # initialize loggers
    logger, logger_metric, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state is not None:
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        for i, o in enumerate(resume_state['optimizers']):
            model.optimizers[i].load_state_dict(o)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        best_metric = resume_state['best_metric']
        best_psnr = best_metric['psnr']
        best_iter = best_metric['iter']
        logger.info(f'best psnr: {best_psnr} from iteration {best_iter}')
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0
        best_metric = {'iter': 0}
        for task, metrics in opt['val']['metrics'].items():
            for k, v in metrics.items():
                best_metric[k] = 0.
        if 'lpips' in best_metric.keys():
            best_metric['lpips'] = float('inf')

    lpips_nets = []
    if opt.get('val') is not None and opt['val'].get('metrics') is not None:
        for task_metrics in opt['val']['metrics'].values():
            if task_metrics is None:
                continue
            for metric_opt in task_metrics.values():
                if metric_opt.get('type') == 'calculate_lpips':
                    lpips_nets.append(metric_opt.get('func_net', 'vgg'))
    lpips_nets = sorted(set(lpips_nets))

    # Optional: preload configured LPIPS weights to avoid a long stall at the first validation.
    if lpips_nets:
        lpips_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if opt.get('dist', False) and dist.is_initialized():
            if opt['rank'] == 0:
                try:
                    preload_lpips_models(lpips_nets, device=lpips_device)
                    torch.cuda.synchronize()
                    logger.info(f"Preloaded LPIPS nets on rank0: {', '.join(lpips_nets)}.")
                except Exception as e:
                    logger.warning(f'LPIPS preload skipped/failed: {e}')
            dist.barrier()
        else:
            try:
                preload_lpips_models(lpips_nets, device=lpips_device)
                torch.cuda.synchronize()
                logger.info(f"Preloaded LPIPS nets (single GPU): {', '.join(lpips_nets)}.")
            except Exception as e:
                logger.warning(f'LPIPS preload skipped/failed: {e}')

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    stage_iter_boundaries = np.cumsum(iters)
    stage_change_logged = [False] * len(stage_iter_boundaries)

    scale = opt['scale']

    epoch = start_epoch

    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            current_stage_index = int(np.searchsorted(stage_iter_boundaries, current_iter, side='left'))
            if current_stage_index >= len(stage_iter_boundaries):
                current_stage_index = len(stage_iter_boundaries) - 1

            mini_gt_size = mini_gt_sizes[current_stage_index]
            mini_batch_size = mini_batch_sizes[current_stage_index]

            if not stage_change_logged[current_stage_index]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(
                    mini_gt_size, mini_batch_size * torch.cuda.device_count()))
                stage_change_logged[current_stage_index] = True

            if 'segment_gt' in train_data:
                lq = train_data['lq']
                denoise_gt = train_data['denoise_gt']
                segment_gt = train_data['segment_gt']

                if mini_batch_size < batch_size:
                    indices = random.sample(
                        range(0, batch_size), k=mini_batch_size)
                    lq = lq[indices]
                    denoise_gt = denoise_gt[indices]
                    segment_gt = segment_gt[indices]

                if mini_gt_size < gt_size:
                    x0 = int((gt_size - mini_gt_size) * random.random())
                    y0 = int((gt_size - mini_gt_size) * random.random())
                    x1 = x0 + mini_gt_size
                    y1 = y0 + mini_gt_size
                    lq = lq[:, :, x0:x1, y0:y1]
                    denoise_gt = denoise_gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]
                    segment_gt = segment_gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]
                model.feed_train_data({'lq': lq, 'denoise_gt': denoise_gt, 'segment_gt': segment_gt})
            else:
                lq = train_data['lq']
                denoise_gt = train_data['denoise_gt']

                if mini_batch_size < batch_size:
                    indices = random.sample(
                        range(0, batch_size), k=mini_batch_size)
                    lq = lq[indices]
                    denoise_gt = denoise_gt[indices]

                if mini_gt_size < gt_size:
                    x0 = int((gt_size - mini_gt_size) * random.random())
                    y0 = int((gt_size - mini_gt_size) * random.random())
                    x1 = x0 + mini_gt_size
                    y1 = y0 + mini_gt_size
                    lq = lq[:, :, x0:x1, y0:y1]
                    denoise_gt = denoise_gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]
                model.feed_train_data({'lq': lq, 'denoise_gt': denoise_gt})

            model.optimize_parameters(current_iter)
            if tb_logger and opt['rank'] == 0:
                tb_logger.add_scalars('losses', model.get_current_log(), current_iter)

            iter_time = time.time() - iter_time
            
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                if opt['rank'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_time, 'data_time': data_time})
                    log_vars.update(model.get_current_log())
                    msg_logger(log_vars)

            save_freq = opt['logger']['save_checkpoint_freq']
            if current_iter % save_freq == 0:
                if opt['rank'] == 0:
                    logger.info('Saving models and training states.')
                    model.save(epoch, current_iter, best_metric=best_metric)
                if opt.get('dist', False) and dist.is_initialized():
                    dist.barrier()

            if (opt.get('val') is not None) and (current_iter % opt['val']['val_freq'] == 0):
                if opt['rank'] == 0:
                    rgb2bgr = opt['val'].get('rgb2bgr', True)
                    use_image = opt['val'].get('use_image', True)
                    current_metric = model.validation(
                        val_loader, current_iter, tb_logger,
                        opt['val']['save_img'], rgb2bgr, use_image)
                    if not isinstance(current_metric, dict):
                        raise TypeError(f"validation() should return a dict on rank0, got {type(current_metric)}")

                    # log current metric to csv
                    logger_metric.info(f'current iteration:{current_iter},{current_metric}')

                    # update best metric (lpips: lower is better)
                    if 'lpips' in current_metric and best_metric.get('lpips', float('inf')) > current_metric['lpips']:
                        best_metric['iter'] = current_iter
                        best_metric.update(current_metric)
                        model.save_best(best_metric)

                    if tb_logger:
                        tb_logger.add_scalar('metrics/best_iter', best_metric.get('iter', 0), current_iter)
                        if 'lpips' in best_metric:
                            tb_logger.add_scalar('metrics/best_lpips', best_metric['lpips'], current_iter)
                        for k, v in current_metric.items():
                            tb_logger.add_scalar(f'metrics/{k}', v, current_iter)
                if opt.get('dist', False) and dist.is_initialized():
                    dist.barrier()

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        epoch += 1

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    if opt['rank'] == 0:
        logger.info('Save the latest model.')
        model.save(epoch=-1, current_iter=-1)
        if tb_logger:
            tb_logger.close()
    if opt.get('dist', False) and dist.is_initialized():
        dist.barrier()


if __name__ == '__main__':
    main()
