import glob
import os
import random
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import metrics as metric_module
import models.losses as loss_module
from models.base_model import BaseModel
from models.archs.jesformer import JESFormer
from utils import get_root_logger, imwrite, tensor2img


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity
        self.augments = [self.mixup]

    def mixup(self, *args):
        output = []
        lam = self.dist.rsample((1, 1)).item()
        r_index = torch.randperm(args[0][0].size(0)).to(self.device)
        for input_ in args[0]:
            output.append(lam * input_ + (1 - lam) * input_[r_index, :])
        return output

    def __call__(self, *args):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            return self.augments[augment](args) if augment < len(self.augments) else args
        augment = random.randint(0, len(self.augments) - 1)
        return self.augments[augment](args)


class ImageEnhancementModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        network_opt = deepcopy(opt['network_g'])
        network_opt.pop('type', None)
        network_opt['is_train'] = self.is_train
        self.net_g = JESFormer(**network_opt)
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            ema_network_opt = deepcopy(self.opt['network_g'])
            ema_network_opt.pop('type', None)
            self.net_g_ema = JESFormer(**ema_network_opt).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        opt_loss = train_opt['loss_opt']
        self.cri_denoise = nn.ModuleDict()
        self.cri_segment = nn.ModuleDict()
        for task, losses in opt_loss.items():
            registry = self.cri_denoise if task == 'denoise' else self.cri_segment
            for loss_type, opt_ in losses.items():
                cri_cls = getattr(loss_module, loss_type)
                registry[loss_type] = cri_cls(**opt_).to(self.device)
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = [value for _, value in self.net_g.named_parameters() if value.requires_grad]
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.denoise_gt = data['denoise_gt'].to(self.device)
        self.segment_gt = data['segment_gt'].to(self.device)
        if self.mixing_flag:
            self.denoise_gt, self.segment_gt, self.lq = self.mixing_augmentation(self.denoise_gt, self.segment_gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'denoise_gt' in data and 'segment_gt' in data:
            self.denoise_gt = data['denoise_gt'].to(self.device)
            self.segment_gt = data['segment_gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        del current_iter
        self.optimizer_g.zero_grad()
        denoised_preds, segmented_preds = self.net_g(self.lq)
        if not isinstance(denoised_preds, list):
            denoised_preds = [denoised_preds]
        if not isinstance(segmented_preds, list):
            segmented_preds = [segmented_preds]
        self.output1, self.output2 = denoised_preds[-1], segmented_preds[-1]

        denoise_loss = 0.0
        segment_loss = 0.0
        denoise_loss_dict = OrderedDict()
        for pred in denoised_preds:
            for loss_type, func in self.cri_denoise.items():
                loss = func(pred, self.denoise_gt)
                denoise_loss += loss
                denoise_loss_dict[loss_type] = loss
        for pred in segmented_preds:
            for _, func in self.cri_segment.items():
                segment_loss += func(pred, self.segment_gt)
        total_loss = denoise_loss + segment_loss
        total_loss.backward()

        loss_dict = {'denoise_loss': denoise_loss, 'segment_loss': segment_loss, 'total_loss': total_loss}
        loss_dict.update(denoise_loss_dict)
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def image_test(self, model, threshold):
        model.eval()
        factor = 8
        h, w = self.lq.shape[2], self.lq.shape[3]
        with torch.no_grad():
            if h <= 2048 and w <= 2048:
                height = ((h + factor) // factor) * factor
                width = ((w + factor) // factor) * factor
                padh = height - h if h % factor != 0 else 0
                padw = width - w if w % factor != 0 else 0
                test_input = F.pad(self.lq, (0, padw, 0, padh), 'reflect')
                denoise_pred, segment_pred = model(test_input)
                denoise_pred = denoise_pred[:, :, :h, :w]
                segment_pred = segment_pred[:, :, :h, :w]
            else:
                height = (h + 2 * factor) // (2 * factor) * (2 * factor)
                width = (w + 2 * factor) // (2 * factor) * (2 * factor)
                padh = height - h if h % (2 * factor) != 0 else 0
                padw = width - w if w % (2 * factor) != 0 else 0
                test_input = F.pad(self.lq, (0, padw, 0, padh), 'reflect')
                input_1 = test_input[:, :, :, 1::2]
                input_2 = test_input[:, :, :, 0::2]
                denoise_pred_1, segment_pred_1 = model(input_1)
                denoise_pred_2, segment_pred_2 = model(input_2)
                denoise_pred = torch.zeros_like(test_input)
                segment_pred = torch.zeros_like(test_input)
                denoise_pred[:, :, :, 1::2] = denoise_pred_1
                denoise_pred[:, :, :, 0::2] = denoise_pred_2
                segment_pred[:, :, :, 1::2] = segment_pred_1
                segment_pred[:, :, :, 0::2] = segment_pred_2
                denoise_pred = denoise_pred[:, :, :h, :w]
                segment_pred = segment_pred[:, :, :h, :w]

            if isinstance(denoise_pred, list):
                denoise_pred = denoise_pred[-1]
            if isinstance(segment_pred, list):
                segment_pred = segment_pred[-1]
            if segment_pred.shape[1] == 1:
                segment_pred = (torch.sigmoid(segment_pred) > threshold).float()
            elif segment_pred.shape[1] > 1:
                segment_pred = torch.argmax(torch.softmax(segment_pred, dim=1), dim=1, keepdim=True).float()
        self.output1 = denoise_pred
        self.output2 = segment_pred
        model.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        return 0.0

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for module in self.opt['val']['metrics'].values() for metric in module.keys()}
        threshold = self.opt['val'].get('segment_threshold', 0.5)
        model = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        count = 0

        for _, val_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.image_test(model, threshold)
            visuals = OrderedDict({'lq': self.lq, 'denoise_result': self.output1, 'segment_result': self.output2})
            seg_min_max = (0, self.output2.max().detach().cpu())
            if hasattr(self, 'denoise_gt'):
                visuals['denoise_gt'] = self.denoise_gt
            if hasattr(self, 'segment_gt'):
                visuals['segment_gt'] = self.segment_gt

            denoise_sr_img = tensor2img([visuals['denoise_result']], rgb2bgr=rgb2bgr)
            segment_sr_img = tensor2img([visuals['segment_result']], rgb2bgr=False, min_max=seg_min_max)
            if 'denoise_gt' in visuals:
                denoise_gt_img = tensor2img([visuals['denoise_gt']], rgb2bgr=rgb2bgr)
                del self.denoise_gt
            if 'segment_gt' in visuals:
                segment_gt_img = tensor2img([visuals['segment_gt']], rgb2bgr=False, min_max=seg_min_max)
                del self.segment_gt

            del self.lq
            del self.output1
            del self.output2
            torch.cuda.empty_cache()

            if save_img:
                save_pred1_img_path = osp.join(self.opt['path']['visualization'], img_name, f'denoised_{img_name}_{current_iter}.png')
                save_pred2_img_path = osp.join(self.opt['path']['visualization'], img_name, f'segmented_{img_name}_{current_iter}.png')
                imwrite(denoise_sr_img, save_pred1_img_path)
                imwrite(segment_sr_img, save_pred2_img_path)

            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                denoise_pred, denoise_gt = (denoise_sr_img, denoise_gt_img) if use_image else (visuals['denoise_result'], visuals['denoise_gt'])
                segment_pred, segment_gt = (segment_sr_img, segment_gt_img) if use_image else (visuals['segment_result'], visuals['segment_gt'])
                for task, metrics in opt_metric.items():
                    if metrics is None:
                        continue
                    for name, opt_ in metrics.items():
                        metric_type = opt_.pop('type')
                        if task == 'denoise':
                            self.metric_results[name] += getattr(metric_module, metric_type)(denoise_pred, denoise_gt, **opt_)
                        else:
                            self.metric_results[name] += getattr(metric_module, metric_type)(segment_pred, segment_gt)
            count += 1

        current_metric = {metric: 0.0 for module in self.opt['val']['metrics'].values() for metric in module.keys()}
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= count
                current_metric[metric] = self.metric_results[metric]
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def save(self, epoch, current_iter, **kwargs):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)

    def save_best(self, best_metric, param_key='params'):
        lpips = best_metric['lpips']
        cur_iter = best_metric['iter']
        save_path = os.path.join(self.opt['path']['experiments_root'], f'best_lpips_{lpips:.2f}_{cur_iter}.pth')
        exp_root = self.opt['path']['experiments_root']
        if os.path.exists(save_path):
            return
        for file_name in glob.glob(f'{exp_root}/best_*'):
            os.remove(file_name)
        nets = self.net_g if isinstance(self.net_g, list) else [self.net_g]
        param_keys = param_key if isinstance(param_key, list) else [param_key]
        save_dict = {}
        for net_, param_key_ in zip(nets, param_keys):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict
        torch.save(save_dict, save_path)
