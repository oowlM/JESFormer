import datetime
import logging
import time
from os import path as osp

from utils.distributed import get_dist_info, master_only
from utils.options import dict2str

initialized_logger = {}


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


class MessageLogger:
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    @master_only
    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        message = f'[epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:('
        for value in lrs:
            message += f'{value:.3e},'
        message += ')] '

        if 'time' in log_vars:
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')
            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, time (data): {iter_time:.3f} ({data_time:.3f})] '

        for key, value in log_vars.items():
            message += f'{key}: {value:.4e} '
            if self.use_tb_logger and 'debug' not in self.exp_name and self.tb_logger is not None:
                tag = f'losses/{key}' if key.startswith('l_') else key
                self.tb_logger.add_scalar(tag, value, current_iter)
        self.logger.info(message)



@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=log_dir)


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='jesformer', 
                             log_level=logging.INFO, log_file=log_file)
    log_file = osp.join(opt['path']['log'], f"metric.csv")
    logger_metric = get_root_logger(logger_name='metric',
                                    log_level=logging.INFO, log_file=log_file)
    metric_str = f'iter ({get_time_str()})'
    for k, v in opt['val']['metrics'].items():
        metric_str += f',{k}'
    logger_metric.info(metric_str)

    logger.info(dict2str(opt))

    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['path']['experiments_root'], 'tb_logger'))
    return logger, logger_metric, tb_logger


def _has_file_handler(logger, log_file):
    target_path = osp.abspath(log_file)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and osp.abspath(handler.baseFilename) == target_path:
            return True
    return False


def get_root_logger(logger_name='jesformer', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'jesformer'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        rank, _ = get_dist_info()
        if rank == 0 and log_file is not None and not _has_file_handler(logger, log_file):
            format_str = '' if logger_name == 'metric' else '%(asctime)s %(levelname)s: %(message)s'
            mode = 'a' if logger_name == 'metric' else 'w'
            file_handler = logging.FileHandler(log_file, mode)
            file_handler.setFormatter(logging.Formatter(format_str))
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
        return logger

    format_str = '' if logger_name == 'metric' else '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False

    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    else:
        logger.setLevel(log_level)
        if log_file is not None:
            mode = 'a' if logger_name == 'metric' else 'w'
            file_handler = logging.FileHandler(log_file, mode)
            file_handler.setFormatter(logging.Formatter(format_str))
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)

    initialized_logger[logger_name] = True
    return logger
