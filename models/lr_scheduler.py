import math
from collections import Counter

from torch.optim.lr_scheduler import _LRScheduler


class MultiStepRestartLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, restarts=(0,), restart_weights=(1,), last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch] for group in self.optimizer.param_groups]


class LinearLR(_LRScheduler):
    def __init__(self, optimizer, total_iter, last_epoch=-1):
        self.total_iter = total_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        weight = 1 - self.last_epoch / self.total_iter
        return [weight * group['initial_lr'] for group in self.optimizer.param_groups]


class VibrateLR(_LRScheduler):
    def __init__(self, optimizer, total_iter, last_epoch=-1):
        self.total_iter = total_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        process = self.last_epoch / self.total_iter
        factor = 0.1
        if process < 3 / 8:
            factor = 1 - process * 8 / 3
        elif process < 5 / 8:
            factor = 0.2
        period = self.total_iter // 80
        half_period = period // 2
        t = self.last_epoch % period
        oscillation = t / half_period
        if t >= half_period:
            oscillation = 2 - oscillation
        weight = factor * oscillation
        if self.last_epoch < half_period:
            weight = max(0.1, weight)
        return [weight * group['initial_lr'] for group in self.optimizer.param_groups]

