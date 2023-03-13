import torch
import numpy as np
import datetime
import random
from pathlib import Path
from shutil import copyfile
from torch.backends import cudnn


def setup_seed(seed):
    """
    设置随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = f'Total : {str(total_num / 1000 ** 2)} M, Trainable: {str(trainable_num / 1000 ** 2)} M'
    return info


def timestr():
    return str(datetime.datetime.now().strftime('%Y-%m%d_%H%M'))


def init_settings(args):
    experiment_dir = Path(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)  # 保存实验日志的总目录
    experiment_dir = experiment_dir.joinpath(timestr() + '_' + args.model_name + '_' + str(args.n))
    experiment_dir.mkdir(exist_ok=True)

    setting_dir = experiment_dir.joinpath('setting')
    setting_dir.mkdir(exist_ok=True)

    # 自然图像
    if args.type == 1:
        checkpoints_dir = experiment_dir.joinpath('checkpoints')
        checkpoints_dir.mkdir(exist_ok=True)
        tensorboard_dir = experiment_dir.joinpath('tensorboard')
        tensorboard_dir.mkdir(exist_ok=True)

        copyfile('config/config1.py', str(setting_dir) + '/config1.py')
        return experiment_dir, checkpoints_dir, tensorboard_dir

    # 井像
    elif args.type == 0:
        copyfile('config/config0.py', str(setting_dir) + '/config0.py')

        return experiment_dir
